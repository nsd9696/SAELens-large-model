from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    TrainCoefficientConfig,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
    TrainStepOutput,
)


def calculate_router_entropy(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate the entropy of router logits.
    
    Args:
        router_logits: Router logits tensor of shape [batch_size, num_experts] or [batch_size, seq_len, num_experts]
    
    Returns:
        Entropy value (scalar tensor)
    """
    # Apply softmax to get probabilities
    router_probs = F.softmax(router_logits, dim=-1)
    # Calculate entropy: -sum(p * log(p))
    entropy = -(router_probs * torch.log(router_probs + 1e-10)).sum(dim=-1)
    # Return mean entropy across batch and sequence dimensions
    return entropy.mean()


def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)


class Step(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,  # noqa: ARG004
    ) -> torch.Tensor:
        return (x > threshold).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        threshold_grad = torch.sum(
            -(1.0 / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output,
            dim=0,
        )
        return None, threshold_grad, None


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,  # noqa: ARG004
    ) -> torch.Tensor:
        return (x * (x > threshold)).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold) * grad_output  # We don't apply STE to x input
        threshold_grad = torch.sum(
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None


@dataclass
class JumpReLUSAEConfig(SAEConfig):
    """
    Configuration class for a JumpReLUSAE.
    """

    @override
    @classmethod
    def architecture(cls) -> str:
        return "jumprelu"


class JumpReLUSAE(SAE[JumpReLUSAEConfig]):
    """
    JumpReLUSAE is an inference-only implementation of a Sparse Autoencoder (SAE)
    using a JumpReLU activation. For each unit, if its pre-activation is
    <= threshold, that unit is zeroed out; otherwise, it follows a user-specified
    activation function (e.g., ReLU etc.).

    It implements:
      - initialize_weights: sets up parameters, including a threshold.
      - encode: computes the feature activations using JumpReLU.
      - decode: reconstructs the input from the feature activations.

    The BaseSAE.forward() method automatically calls encode and decode,
    including any error-term processing if configured.
    """

    b_enc: nn.Parameter
    threshold: nn.Parameter

    def __init__(self, cfg: JumpReLUSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self.threshold = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor into the feature space using JumpReLU.
        The threshold parameter determines which units remain active.
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        # 1) Apply the base "activation_fn" from config (e.g., ReLU).
        base_acts = self.activation_fn(hidden_pre)

        # 2) Zero out any unit whose (hidden_pre <= threshold).
        #    We cast the boolean mask to the same dtype for safe multiplication.
        jump_relu_mask = (hidden_pre > self.threshold).to(base_acts.dtype)

        # 3) Multiply the normally activated units by that mask.
        return self.hook_sae_acts_post(base_acts * jump_relu_mask)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode the feature activations back to the input space.
        Follows the same steps as StandardSAE: apply scaling, transform, hook, and optionally reshape.
        """
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @torch.no_grad()
    def fold_W_dec_norm(self):
        """
        Override to properly handle threshold adjustment with W_dec norms.
        When we scale the encoder weights, we need to scale the threshold
        by the same factor to maintain the same sparsity pattern.
        """
        # Save the current threshold before calling parent method
        current_thresh = self.threshold.clone()

        # Get W_dec norms that will be used for scaling (clamped to avoid division by zero)
        W_dec_norms = self.W_dec.norm(dim=-1).clamp(min=1e-8)

        # Call parent implementation to handle W_enc, W_dec, and b_enc adjustment
        super().fold_W_dec_norm()

        # Scale the threshold by the same factor as we scaled b_enc
        # This ensures the same features remain active/inactive after folding
        self.threshold.data = current_thresh * W_dec_norms


@dataclass
class JumpReLUTrainingSAEConfig(TrainingSAEConfig):
    """
    Configuration class for training a JumpReLUTrainingSAE.

    - jumprelu_init_threshold: initial threshold for the JumpReLU activation
    - jumprelu_bandwidth: bandwidth for the JumpReLU activation
    - jumprelu_sparsity_loss_mode: mode for the sparsity loss, either "step" or "tanh". "step" is Google Deepmind's L0 loss, "tanh" is Anthropic's sparsity loss.
    - l0_coefficient: coefficient for the l0 sparsity loss
    - l0_warm_up_steps: number of warm-up steps for the l0 sparsity loss
    - pre_act_loss_coefficient: coefficient for the pre-activation loss. Set to None to disable. Set to 3e-6 to match Anthropic's setup. Default is None.
    - jumprelu_tanh_scale: scale for the tanh sparsity loss. Only relevant for "tanh" sparsity loss mode. Default is 4.0.
    """

    jumprelu_init_threshold: float = 0.01
    jumprelu_bandwidth: float = 0.05
    # step is Google Deepmind, tanh is Anthropic
    jumprelu_sparsity_loss_mode: Literal["step", "tanh"] = "step"
    l0_coefficient: float = 1.0
    l0_warm_up_steps: int = 0

    # anthropic's auxiliary loss to avoid dead features
    pre_act_loss_coefficient: float | None = None

    # only relevant for tanh sparsity loss mode
    jumprelu_tanh_scale: float = 4.0
    
    router_entropy_layer: str | None = None  # model.layers.16.mlp.router
    use_router_entropy: bool = False  # Whether to use router entropy to adjust l0 coefficient
    router_entropy_weight: float = 0.1  # Weight for router entropy adjustment

    @override
    @classmethod
    def architecture(cls) -> str:
        return "jumprelu"


class JumpReLUTrainingSAE(TrainingSAE[JumpReLUTrainingSAEConfig]):
    """
    JumpReLUTrainingSAE is a training-focused implementation of a SAE using a JumpReLU activation.

    Similar to the inference-only JumpReLUSAE, but with:
      - A learnable log-threshold parameter (instead of a raw threshold).
      - A specialized auxiliary loss term for sparsity (L0 or similar).

    Methods of interest include:
    - initialize_weights: sets up W_enc, b_enc, W_dec, b_dec, and log_threshold.
    - encode_with_hidden_pre_jumprelu: runs a forward pass for training.
    - training_forward_pass: calculates MSE and auxiliary losses, returning a TrainStepOutput.
    """

    b_enc: nn.Parameter
    log_threshold: nn.Parameter
    cfg: JumpReLUTrainingSAEConfig  # type: ignore[assignment]
    model: Any = None  # Reference to the model for accessing router activations
    router_entropy_buffer: torch.Tensor | None = None  # Buffer for tracking router entropy

    def __init__(self, cfg: JumpReLUTrainingSAEConfig, use_error_term: bool = False, model: Any = None):
        super().__init__(cfg, use_error_term)

        self.model = model

        # We'll store a bandwidth for the training approach, if needed
        self.bandwidth = cfg.jumprelu_bandwidth

        # In typical JumpReLU training code, we may track a log_threshold:
        self.log_threshold = nn.Parameter(
            torch.ones(self.cfg.d_sae, dtype=self.dtype, device=self.device)
            * np.log(cfg.jumprelu_init_threshold)
        )
        
        # Register buffer for tracking router entropy (for normalization)
        if cfg.use_router_entropy:
            # Check if buffer already exists (e.g., when loading from dict)
            if "router_entropy_buffer" not in self._buffers and not hasattr(self, "router_entropy_buffer"):
                self.register_buffer(
                    "router_entropy_buffer",
                    torch.tensor(0.0, dtype=torch.float32, device=self.W_dec.device),
                )

    @override
    def initialize_weights(self) -> None:
        """
        Initialize parameters like the base SAE, but also add log_threshold.
        """
        super().initialize_weights()
        # Encoder Bias
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

    @property
    def threshold(self) -> torch.Tensor:
        """
        Returns the parameterized threshold > 0 for each unit.
        threshold = exp(log_threshold).
        """
        return torch.exp(self.log_threshold)

    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        # Get router entropy if enabled
        router_entropy = None
        if self.cfg.use_router_entropy and self.cfg.router_entropy_layer is not None and self.model is not None:
            router_entropy = self._get_router_entropy()
            if router_entropy is not None:
                # Update router entropy buffer for normalization
                if self.router_entropy_buffer is not None:
                    # Use exponential moving average
                    lr = 0.01  # Learning rate for entropy tracking
                    self.router_entropy_buffer = (
                        (1 - lr) * self.router_entropy_buffer + lr * router_entropy
                    )

        hidden_pre = sae_in @ self.W_enc + self.b_enc
        feature_acts = JumpReLU.apply(hidden_pre, self.threshold, self.bandwidth)

        return feature_acts, hidden_pre  # type: ignore

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Calculate architecture-specific auxiliary loss terms."""

        threshold = self.threshold
        W_dec_norm = self.W_dec.norm(dim=1)
        
        # Adjust l0 coefficient based on router entropy if enabled
        l0_coefficient = step_input.coefficients["l0"]
        if self.cfg.use_router_entropy and self.router_entropy_buffer is not None:
            base_entropy = self.router_entropy_buffer.item()
            if base_entropy > 0:
                # Get current router entropy
                router_entropy = self._get_router_entropy()
                if router_entropy is not None:
                    entropy_ratio = router_entropy.item() / (base_entropy + 1e-10)
                    # Higher entropy -> more uniform -> increase l0 penalty
                    # Lower entropy -> more concentrated -> decrease l0 penalty
                    l0_coefficient = l0_coefficient * (1.0 + self.cfg.router_entropy_weight * (entropy_ratio - 1.0))
        
        if self.cfg.jumprelu_sparsity_loss_mode == "step":
            l0 = torch.sum(
                Step.apply(hidden_pre, threshold, self.bandwidth),  # type: ignore
                dim=-1,
            )
            l0_loss = (l0_coefficient * l0).mean()
        elif self.cfg.jumprelu_sparsity_loss_mode == "tanh":
            per_item_l0_loss = torch.tanh(
                self.cfg.jumprelu_tanh_scale * feature_acts * W_dec_norm
            ).sum(dim=-1)
            l0_loss = (l0_coefficient * per_item_l0_loss).mean()
        else:
            raise ValueError(
                f"Invalid sparsity loss mode: {self.cfg.jumprelu_sparsity_loss_mode}"
            )
        losses = {"l0_loss": l0_loss}

        if self.cfg.pre_act_loss_coefficient is not None:
            losses["pre_act_loss"] = calculate_pre_act_loss(
                self.cfg.pre_act_loss_coefficient,
                threshold,
                hidden_pre,
                step_input.dead_neuron_mask,
                W_dec_norm,
            )
        return losses
    
    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)
        
        # Log router entropy if available
        if self.router_entropy_buffer is not None:
            output.metrics["router_entropy"] = self.router_entropy_buffer.item()
        
        return output
    
    @torch.no_grad()
    def _get_router_entropy(self) -> torch.Tensor | None:
        """
        Get router entropy from the model's router layer.
        
        Returns:
            Router entropy value or None if not available
        """
        if self.model is None or self.cfg.router_entropy_layer is None:
            return None
        
        try:
            # Try to get router activations from the model's cache
            # This assumes the model has been run with caching enabled
            if hasattr(self.model, "cache") and self.model.cache is not None:
                router_activations = self.model.cache.get(self.cfg.router_entropy_layer, None)
                if router_activations is not None:
                    return calculate_router_entropy(router_activations)
        except Exception as e:
            # If we can't get router entropy, return None
            # This allows training to continue even if router entropy is not available
            pass
        
        return None

    @override
    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        return {
            "l0": TrainCoefficientConfig(
                value=self.cfg.l0_coefficient,
                warm_up_steps=self.cfg.l0_warm_up_steps,
            ),
        }

    @torch.no_grad()
    def fold_W_dec_norm(self):
        """
        Override to properly handle threshold adjustment with W_dec norms.
        """
        # Save the current threshold before we call the parent method
        current_thresh = self.threshold.clone()

        # Get W_dec norms (clamped to avoid division by zero)
        W_dec_norms = self.W_dec.norm(dim=-1).clamp(min=1e-8).unsqueeze(1)

        # Call parent implementation to handle W_enc and W_dec adjustment
        super().fold_W_dec_norm()

        # Fix: Use squeeze() instead of squeeze(-1) to match old behavior
        self.log_threshold.data = torch.log(current_thresh * W_dec_norms.squeeze())

    def process_state_dict_for_saving(self, state_dict: dict[str, Any]) -> None:
        """Convert log_threshold to threshold for saving"""
        if "log_threshold" in state_dict:
            threshold = torch.exp(state_dict["log_threshold"]).detach().contiguous()
            del state_dict["log_threshold"]
            state_dict["threshold"] = threshold

    def process_state_dict_for_loading(self, state_dict: dict[str, Any]) -> None:
        """Convert threshold to log_threshold for loading"""
        if "threshold" in state_dict:
            threshold = state_dict["threshold"]
            del state_dict["threshold"]
            state_dict["log_threshold"] = torch.log(threshold).detach().contiguous()


def calculate_pre_act_loss(
    pre_act_loss_coefficient: float,
    threshold: torch.Tensor,
    hidden_pre: torch.Tensor,
    dead_neuron_mask: torch.Tensor | None,
    W_dec_norm: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate Anthropic's pre-activation loss, except we only calculate this for latents that are actually dead.
    """
    if dead_neuron_mask is None or not dead_neuron_mask.any():
        return hidden_pre.new_tensor(0.0)
    per_item_loss = (
        (threshold - hidden_pre).relu() * dead_neuron_mask * W_dec_norm
    ).sum(dim=-1)
    return pre_act_loss_coefficient * per_item_loss.mean()
