from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override

from sae_lens.saes.jumprelu_sae import JumpReLUSAEConfig
from sae_lens.saes.sae import SAEConfig, TrainStepInput, TrainStepOutput
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig


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


class BatchTopK(nn.Module):
    """BatchTopK activation function with optional router entropy adjustment"""

    def __init__(
        self,
        k: float,
        use_router_entropy: bool = False,
        router_entropy_weight: float = 1.0,
        base_entropy: float = None,
    ):
        super().__init__()
        self.k = k
        self.use_router_entropy = use_router_entropy
        self.router_entropy_weight = router_entropy_weight
        self.base_entropy = base_entropy

    def forward(self, x: torch.Tensor, router_entropy: torch.Tensor | None = None) -> torch.Tensor:
        acts = x.relu()
        flat_acts = acts.flatten()
        # Calculate total number of samples across all non-feature dimensions
        num_samples = acts.shape[:-1].numel()
        
        # Adjust k based on router entropy if enabled
        effective_k = self.k
        if self.use_router_entropy and router_entropy is not None:
            if self.base_entropy is not None:
                # Normalize entropy relative to base entropy
                # Higher entropy -> more uniform distribution -> increase k
                # Lower entropy -> more concentrated -> decrease k
                entropy_ratio = router_entropy / (self.base_entropy + 1e-10)
                # Scale k based on entropy ratio
                effective_k = self.k * (1.0 + self.router_entropy_weight * (entropy_ratio - 1.0))
            else:
                # If no base entropy, use raw entropy value
                # This assumes entropy is in a reasonable range (e.g., 0-10)
                effective_k = self.k * (1.0 + self.router_entropy_weight * router_entropy.item())
        
        acts_topk_flat = torch.topk(flat_acts, int(effective_k * num_samples), dim=-1)
        return (
            torch.zeros_like(flat_acts)
            .scatter(-1, acts_topk_flat.indices, acts_topk_flat.values)
            .reshape(acts.shape)
        )


@dataclass
class BatchTopKTrainingSAEConfig(TopKTrainingSAEConfig):
    """
    Configuration class for training a BatchTopKTrainingSAE.

    BatchTopK SAEs maintain k active features on average across the entire batch,
    rather than enforcing k features per sample like standard TopK SAEs. During training,
    the SAE learns a global threshold that is updated based on the minimum positive
    activation value. After training, BatchTopK SAEs are saved as JumpReLU SAEs.

    Args:
        k (float): Average number of features to keep active across the batch. Unlike
            standard TopK SAEs where k is an integer per sample, this is a float
            representing the average number of active features across all samples in
            the batch. Defaults to 100.
        topk_threshold_lr (float): Learning rate for updating the global topk threshold.
            The threshold is updated using an exponential moving average of the minimum
            positive activation value. Defaults to 0.01.
        aux_loss_coefficient (float): Coefficient for the auxiliary loss that encourages
            dead neurons to learn useful features. Inherited from TopKTrainingSAEConfig.
            Defaults to 1.0.
        rescale_acts_by_decoder_norm (bool): Treat the decoder as if it was already normalized.
            Inherited from TopKTrainingSAEConfig. Defaults to True.
        decoder_init_norm (float | None): Norm to initialize decoder weights to.
            Inherited from TrainingSAEConfig. Defaults to 0.1.
        d_in (int): Input dimension (dimensionality of the activations being encoded).
            Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
            Defaults to "float32".
        device (str): Device to place the SAE on. Inherited from SAEConfig.
            Defaults to "cpu".
    """

    k: float = 100  # type: ignore[assignment]
    topk_threshold_lr: float = 0.01
    router_entropy_layer: str | None = None  # model.layers.16.mlp.router
    use_router_entropy: bool = False  # Whether to use router entropy to adjust k
    router_entropy_weight: float = 0.1  # Weight for router entropy adjustment

    @override
    @classmethod
    def architecture(cls) -> str:
        return "batchtopk"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        return JumpReLUSAEConfig


class BatchTopKTrainingSAE(TopKTrainingSAE):
    """
    Global Batch TopK Training SAE

    This SAE will maintain the k on average across the batch, rather than enforcing the k per-sample as in standard TopK.

    BatchTopK SAEs are saved as JumpReLU SAEs after training.
    """

    topk_threshold: torch.Tensor
    cfg: BatchTopKTrainingSAEConfig  # type: ignore[assignment]
    model: Any = None  # Reference to the model for accessing router activations
    router_entropy_buffer: torch.Tensor | None = None  # Buffer for tracking router entropy

    def __init__(self, cfg: BatchTopKTrainingSAEConfig, use_error_term: bool = False, model: Any = None):
        super().__init__(cfg, use_error_term)

        self.model = model

        self.register_buffer(
            "topk_threshold",
            # use double precision as otherwise we can run into numerical issues
            torch.tensor(0.0, dtype=torch.double, device=self.W_dec.device),
        )
        
        # Register buffer for tracking router entropy (for normalization)
        if cfg.use_router_entropy:
            # Check if buffer already exists (e.g., when loading from dict)
            # Check both _buffers dict and hasattr to handle all cases
            if "router_entropy_buffer" not in self._buffers and not hasattr(self, "router_entropy_buffer"):
                self.register_buffer(
                    "router_entropy_buffer",
                    torch.tensor(0.0, dtype=torch.float32, device=self.W_dec.device),
                )

    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        base_entropy = self.router_entropy_buffer.item() if self.router_entropy_buffer is not None else None
        return BatchTopK(
            self.cfg.k,
            use_router_entropy=self.cfg.use_router_entropy,
            router_entropy_weight=self.cfg.router_entropy_weight,
            base_entropy=base_entropy,
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Similar to the base training method: calculate pre-activations, then apply BatchTopK.
        Override to pass router entropy to the activation function.
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

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

        # Apply the BatchTopK activation function with router entropy
        if isinstance(self.activation_fn, BatchTopK) and router_entropy is not None:
            feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre, router_entropy))
        else:
            feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return feature_acts, hidden_pre

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)
        self.update_topk_threshold(output.feature_acts)
        output.metrics["topk_threshold"] = self.topk_threshold
        
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

    @torch.no_grad()
    def update_topk_threshold(self, acts_topk: torch.Tensor) -> None:
        positive_mask = acts_topk > 0
        lr = self.cfg.topk_threshold_lr
        # autocast can cause numerical issues with the threshold update
        with torch.autocast(self.topk_threshold.device.type, enabled=False):
            if positive_mask.any():
                min_positive = (
                    acts_topk[positive_mask].min().to(self.topk_threshold.dtype)
                )
                self.topk_threshold = (1 - lr) * self.topk_threshold + lr * min_positive

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        super().process_state_dict_for_saving_inference(state_dict)
        # turn the topk threshold into jumprelu threshold
        topk_threshold = state_dict.pop("topk_threshold").item()
        state_dict["threshold"] = torch.ones_like(self.b_enc) * topk_threshold
