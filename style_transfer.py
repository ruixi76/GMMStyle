"""Style transfer modules for domain adaptation."""
import torch
import torch.nn as nn
from typing import Dict, Optional


class PixelStyleTransfer(nn.Module):
    """
    Pixel-level style transfer module.
    
    Core formula: I' = (I - μ_s)/σ_s * σ_t + μ_t
    where μ and σ are channel-level statistics, not GMM covariance matrices.
    """
    
    def __init__(
        self,
        num_components: int = 5,
        eps: float = 1e-6,
        alpha: float = 0.5
    ):
        super().__init__()
        self.num_components = num_components
        self.eps = eps
        self.alpha = alpha
    
    def forward(
        self,
        source_images: torch.Tensor,
        responsibilities: torch.Tensor,
        source_stats: Dict[str, torch.Tensor],
        target_stats: Dict[str, torch.Tensor],
        alpha: Optional[float] = None
    ) -> torch.Tensor:
        """
        Batch-level style transfer grouped by GMM components.
        
        Args:
            source_images: Source domain images [B, C, H, W]
            responsibilities: Responsibility matrix [B, H, W, K]
            source_stats: Source component stats {'means': [k, C], 'stds': [k, C]}
            target_stats: Target component stats {'means': [k, C], 'stds': [k, C]}
            alpha: Style transfer strength (0.0-1.0)
            
        Returns:
            styled_images: Style-transferred images [B, C, H, W]
        """
        if alpha is None:
            alpha = self.alpha
        
        bsz, channels, h, w = source_images.shape
        num_components = responsibilities.shape[-1]
        
        if num_components != self.num_components:
            raise ValueError(
                f"responsibilities components ({num_components}) != "
                f"num_components ({self.num_components})"
            )
        
        # [B, H, W, C]
        source_nhwc = source_images.permute(0, 2, 3, 1)
        
        # Broadcast to [1, 1, 1, K, C]
        mu_s = source_stats['means'].view(
            1, 1, 1, self.num_components, channels
        )
        sigma_s = source_stats['stds'].view(
            1, 1, 1, self.num_components, channels
        )
        mu_t = target_stats['means'].view(
            1, 1, 1, self.num_components, channels
        )
        sigma_t = target_stats['stds'].view(
            1, 1, 1, self.num_components, channels
        )
        
        # Expand pixels to [B, H, W, 1, C] for candidate computation
        source_expand = source_nhwc.unsqueeze(3)
        candidates = (
            (source_expand - mu_s) / (sigma_s + self.eps)
        ) * sigma_t + mu_t
        
        # Soft assignment fusion: x_style = Σ_k γ_nk * x_hat_nk
        gamma = responsibilities.unsqueeze(-1)  # [B, H, W, K, 1]
        styled_nhwc = torch.sum(gamma * candidates, dim=3)  # [B, H, W, C]
        
        # Content preservation blend: x_final = α*x_style + (1-α)*x_source
        styled_nhwc = alpha * styled_nhwc + (1.0 - alpha) * source_nhwc
        
        styled_images = styled_nhwc.permute(0, 3, 1, 2)
        styled_images = torch.clamp(styled_images, 0.0, 1.0)
        
        return styled_images


class FeatureStyleTransfer(nn.Module):
    """
    Feature-level WCT (Whitening and Coloring Transform) style transfer.
    Uses component-wise affine transformation with soft assignment fusion.
    """
    
    def __init__(self, num_components: int = 5, eps: float = 1e-5):
        super().__init__()
        self.num_components = num_components
        self.eps = eps
    
    def _safe_wct_matrix(
        self, cov_s: torch.Tensor, cov_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute safe WCT transform matrix using eigendecomposition.
        
        Args:
            cov_s: Source covariance [D, D]
            cov_t: Target covariance [D, D]
            
        Returns:
            WCT transform matrix [D, D]
        """
        eye = torch.eye(
            cov_s.size(0), device=cov_s.device, dtype=cov_s.dtype
        )
        cov_s = cov_s + self.eps * eye
        cov_t = cov_t + self.eps * eye
        
        # Eigendecomposition
        eval_s, evec_s = torch.linalg.eigh(cov_s)
        eval_t, evec_t = torch.linalg.eigh(cov_t)
        
        # Clamp eigenvalues for numerical stability
        eval_s = torch.clamp(eval_s, min=self.eps)
        eval_t = torch.clamp(eval_t, min=self.eps)
        
        # Whitening and coloring transforms
        ds_inv_sqrt = torch.diag(torch.rsqrt(eval_s))
        dt_sqrt = torch.diag(torch.sqrt(eval_t))
        
        whiten = evec_s @ ds_inv_sqrt @ evec_s.T
        color = evec_t @ dt_sqrt @ evec_t.T
        
        return color @ whiten
    
    def forward(
        self,
        source_features: torch.Tensor,
        responsibilities: torch.Tensor,
        source_stats: Dict[str, torch.Tensor],
        target_stats: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply feature-level style transfer using WCT.
        
        Args:
            source_features: [B, D]
            responsibilities: [B, K]
            source_stats: {'means': [K, D], 'covariances': [K, D, D]}
            target_stats: {'means': [K, D], 'covariances': [K, D, D]}
            
        Returns:
            stylized_features: [B, D]
        """
        batch_size, feat_dim = source_features.shape
        
        if responsibilities.shape[-1] != self.num_components:
            raise ValueError(
                "Feature responsibilities shape mismatch with num_components"
            )
        
        # Compute candidate transformations for each component
        candidates = []
        for k in range(self.num_components):
            mu_s = source_stats['means'][k]
            mu_t = target_stats['means'][k]
            cov_s = source_stats['covariances'][k]
            cov_t = target_stats['covariances'][k]
            
            # Compute WCT matrix
            w_k = self._safe_wct_matrix(cov_s, cov_t)
            b_k = mu_t - w_k @ mu_s
            
            # Apply affine transform
            xk = source_features @ w_k.T + b_k.view(1, feat_dim)
            candidates.append(xk)
        
        # Stack candidates: [B, K, D]
        candidates = torch.stack(candidates, dim=1)
        
        # Soft assignment fusion
        gamma = responsibilities.unsqueeze(-1)
        stylized_features = torch.sum(gamma * candidates, dim=1)
        
        return stylized_features
