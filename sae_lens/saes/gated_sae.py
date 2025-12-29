from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from numpy.typing import NDArray
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


@dataclass
class GatedSAEConfig(SAEConfig):
    """
    Configuration class for a GatedSAE.
    """

    @override
    @classmethod
    def architecture(cls) -> str:
        return "gated"


class GatedSAE(SAE[GatedSAEConfig]):
    """
    GatedSAE is an inference-only implementation of a Sparse Autoencoder (SAE)
    using a gated linear encoder and a standard linear decoder.
    """

    b_gate: nn.Parameter
    b_mag: nn.Parameter
    r_mag: nn.Parameter

    def __init__(self, cfg: GatedSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        # Ensure b_enc does not exist for the gated architecture
        self.b_enc = None

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_gated(self)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor into the feature space using a gated encoder.
        This must match the original encode_gated implementation from SAE class.
        """
        # Preprocess the SAE input (casting type, applying hooks, normalization)
        sae_in = self.process_sae_in(x)

        # Gating path exactly as in original SAE.encode_gated
        gating_pre_activation = sae_in @ self.W_enc + self.b_gate
        active_features = (gating_pre_activation > 0).to(self.dtype)

        # Magnitude path (weight sharing with gated encoder)
        magnitude_pre_activation = self.hook_sae_acts_pre(
            sae_in @ (self.W_enc * self.r_mag.exp()) + self.b_mag
        )
        feature_magnitudes = self.activation_fn(magnitude_pre_activation)

        # Combine gating and magnitudes
        return self.hook_sae_acts_post(active_features * feature_magnitudes)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode the feature activations back into the input space:
          1) Apply optional finetuning scaling.
          2) Linear transform plus bias.
          3) Run any reconstruction hooks and out-normalization if configured.
          4) If the SAE was reshaping hook_z activations, reshape back.
        """
        # 1) optional finetuning scaling
        # 2) linear transform
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        # 3) hooking and normalization
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        # 4) reshape if needed (hook_z)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @torch.no_grad()
    def fold_W_dec_norm(self):
        """Override to handle gated-specific parameters."""
        W_dec_norms = self.W_dec.norm(dim=-1).clamp(min=1e-8).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T

        # Gated-specific parameters need special handling
        # r_mag doesn't need scaling since W_enc scaling is sufficient for magnitude path
        self.b_gate.data = self.b_gate.data * W_dec_norms.squeeze()
        self.b_mag.data = self.b_mag.data * W_dec_norms.squeeze()


@dataclass
class GatedTrainingSAEConfig(TrainingSAEConfig):
    """
    Configuration class for training a GatedTrainingSAE.
    """

    l1_coefficient: float = 1.0
    l1_warm_up_steps: int = 0
    router_entropy_layer: str | None = None  # model.layers.16.mlp.router
    use_router_entropy: bool = False  # Whether to use router entropy to adjust l1 coefficient
    router_entropy_weight: float = 0.1  # Weight for router entropy adjustment

    @override
    @classmethod
    def architecture(cls) -> str:
        return "gated"


class GatedTrainingSAE(TrainingSAE[GatedTrainingSAEConfig]):
    """
    GatedTrainingSAE is a concrete implementation of BaseTrainingSAE for the "gated" SAE architecture.
    It implements:
      - initialize_weights: sets up gating parameters (as in GatedSAE) plus optional training-specific init.
      - encode: calls encode_with_hidden_pre (standard training approach).
      - decode: linear transformation + hooking, same as GatedSAE or StandardTrainingSAE.
      - encode_with_hidden_pre: gating logic.
      - calculate_aux_loss: includes an auxiliary reconstruction path and gating-based sparsity penalty.
      - training_forward_pass: calls encode_with_hidden_pre, decode, and sums up MSE + gating losses.
    """

    b_gate: nn.Parameter  # type: ignore
    b_mag: nn.Parameter  # type: ignore
    r_mag: nn.Parameter  # type: ignore
    cfg: GatedTrainingSAEConfig  # type: ignore[assignment]
    model: Any = None  # Reference to the model for accessing router activations
    router_entropy_buffer: torch.Tensor | None = None  # Buffer for tracking router entropy

    def __init__(self, cfg: GatedTrainingSAEConfig, use_error_term: bool = False, model: Any = None):
        if use_error_term:
            raise ValueError(
                "GatedSAE does not support `use_error_term`. Please set `use_error_term=False`."
            )
        super().__init__(cfg, use_error_term)
        
        self.model = model
        
        # Register buffer for tracking router entropy (for normalization)
        if cfg.use_router_entropy:
            # Check if buffer already exists (e.g., when loading from dict)
            if "router_entropy_buffer" not in self._buffers and not hasattr(self, "router_entropy_buffer"):
                self.register_buffer(
                    "router_entropy_buffer",
                    torch.tensor(0.0, dtype=torch.float32, device=self.W_dec.device),
                )

    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_gated(self)

    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gated forward pass with pre-activation (for training).
        """
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

        # Gating path
        gating_pre_activation = sae_in @ self.W_enc + self.b_gate
        active_features = (gating_pre_activation > 0).to(self.dtype)

        # Magnitude path
        magnitude_pre_activation = sae_in @ (self.W_enc * self.r_mag.exp()) + self.b_mag
        magnitude_pre_activation = self.hook_sae_acts_pre(magnitude_pre_activation)

        feature_magnitudes = self.activation_fn(magnitude_pre_activation)

        # Combine gating path and magnitude path
        feature_acts = self.hook_sae_acts_post(active_features * feature_magnitudes)

        # Return both the final feature activations and the pre-activation (for logging or penalty)
        return feature_acts, magnitude_pre_activation

    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Re-center the input if apply_b_dec_to_input is set
        sae_in_centered = step_input.sae_in - (
            self.b_dec * self.cfg.apply_b_dec_to_input
        )

        # The gating pre-activation (pi_gate) for the auxiliary path
        pi_gate = sae_in_centered @ self.W_enc + self.b_gate
        pi_gate_act = torch.relu(pi_gate)

        # Adjust l1 coefficient based on router entropy if enabled
        l1_coefficient = step_input.coefficients["l1"]
        if self.cfg.use_router_entropy and self.router_entropy_buffer is not None:
            base_entropy = self.router_entropy_buffer.item()
            if base_entropy > 0:
                # Get current router entropy
                router_entropy = self._get_router_entropy()
                if router_entropy is not None:
                    entropy_ratio = router_entropy.item() / (base_entropy + 1e-10)
                    # Higher entropy -> more uniform -> increase l1 penalty
                    # Lower entropy -> more concentrated -> decrease l1 penalty
                    l1_coefficient = l1_coefficient * (1.0 + self.cfg.router_entropy_weight * (entropy_ratio - 1.0))

        # L1-like penalty scaled by W_dec norms
        l1_loss = (
            l1_coefficient
            * torch.sum(pi_gate_act * self.W_dec.norm(dim=1), dim=-1).mean()
        )

        # Aux reconstruction: reconstruct x purely from gating path
        via_gate_reconstruction = pi_gate_act @ self.W_dec + self.b_dec
        aux_recon_loss = (
            (via_gate_reconstruction - step_input.sae_in).pow(2).sum(dim=-1).mean()
        )

        # Return both losses separately
        return {"l1_loss": l1_loss, "auxiliary_reconstruction_loss": aux_recon_loss}
    
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

    def log_histograms(self) -> dict[str, NDArray[Any]]:
        """Log histograms of the weights and biases."""
        b_gate_dist = self.b_gate.detach().float().cpu().numpy()
        b_mag_dist = self.b_mag.detach().float().cpu().numpy()
        return {
            **super().log_histograms(),
            "weights/b_gate": b_gate_dist,
            "weights/b_mag": b_mag_dist,
        }

    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        return {
            "l1": TrainCoefficientConfig(
                value=self.cfg.l1_coefficient,
                warm_up_steps=self.cfg.l1_warm_up_steps,
            ),
        }

    @torch.no_grad()
    def fold_W_dec_norm(self):
        """Override to handle gated-specific parameters."""
        W_dec_norms = self.W_dec.norm(dim=-1).clamp(min=1e-8).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T

        # Gated-specific parameters need special handling
        # r_mag doesn't need scaling since W_enc scaling is sufficient for magnitude path
        self.b_gate.data = self.b_gate.data * W_dec_norms.squeeze()
        self.b_mag.data = self.b_mag.data * W_dec_norms.squeeze()


def _init_weights_gated(
    sae: SAE[GatedSAEConfig] | TrainingSAE[GatedTrainingSAEConfig],
) -> None:
    sae.b_gate = nn.Parameter(
        torch.zeros(sae.cfg.d_sae, dtype=sae.dtype, device=sae.device)
    )
    # Ensure r_mag is initialized to zero as in original
    sae.r_mag = nn.Parameter(
        torch.zeros(sae.cfg.d_sae, dtype=sae.dtype, device=sae.device)
    )
    sae.b_mag = nn.Parameter(
        torch.zeros(sae.cfg.d_sae, dtype=sae.dtype, device=sae.device)
    )
