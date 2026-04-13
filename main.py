"""Main training script for Pixel-level GMM Domain Adaptation."""
import torch
import random
import numpy as np
from config import Config
from datasets import get_dataloaders
from model import DomainAdapter
from domain_adapter import GMMStyleDomainAdapter
from pixel_gmm import PixelGaussianMixture


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_target_pixels(
    target_loader,
    device: torch.device,
    num_batches: int = 1
) -> torch.Tensor:
    """
    Collect pixel samples from multiple batches for BIC evaluation.
    
    Args:
        target_loader: Target domain DataLoader
        device: Computing device (CPU or GPU)
        num_batches: Number of batches to sample
        
    Returns:
        Tensor of shape [N, 3] containing N RGB pixel values
    """
    pixel_batches = []
    for batch_idx, (images, _, _, _) in enumerate(target_loader):
        if batch_idx >= max(1, num_batches):
            break
        images = images.to(device)
        # [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
        pixels = images.permute(0, 2, 3, 1).reshape(-1, 3)
        pixel_batches.append(pixels)
    
    return torch.cat(pixel_batches, dim=0)


def auto_select_num_gaussians(config, target_loader) -> None:
    """
    Automatically select optimal number of Gaussian components using BIC.
    
    Args:
        config: Configuration object
        target_loader: Target domain DataLoader
    """
    device = torch.device(config.device)
    target_pixels = collect_target_pixels(
        target_loader,
        device,
        num_batches=config.bic_num_batches
    )
    
    print(f"\n[BIC] Evaluating K candidates: {config.bic_k_candidates}")
    best_k = config.num_gaussians
    best_bic = float('inf')
    
    for k in config.bic_k_candidates:
        gmm = PixelGaussianMixture(
            num_components=k,
            feature_dim=3,
            device=config.device,
            covariance_type=config.covariance_type,
            max_iters=config.bic_em_iters,
            tol=config.gmm_convergence_threshold
        )
        bic = gmm.get_bic(target_pixels, max_iters=config.bic_em_iters)
        print(f"[BIC] K={k}, BIC={bic:.2f}")
        
        if bic < best_bic:
            best_bic = bic
            best_k = k
    
    config.num_gaussians = best_k
    print(f"[BIC] Selected K={best_k} (min BIC={best_bic:.2f})")


def main():
    """Main training entry point."""
    # 1. Parse configuration
    config = Config().parse()
    set_seed(config.seed)
    
    print(f"Using device: {config.device}")
    if 'cuda' in str(config.device):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. Load datasets
    print("\nLoading datasets...")
    source_train_loader, source_val_loader, target_train_loader, target_test_loader = get_dataloaders(config)
    
    # 3. Auto-select number of Gaussians if enabled
    if getattr(config, 'auto_select_k', False):
        auto_select_num_gaussians(config, target_train_loader)
    
    # 4. Create model
    print("\nCreating model...")
    model = DomainAdapter(config)
    
    # 5. Create trainer
    print("Creating domain adapter...")
    trainer = GMMStyleDomainAdapter(model, config)
    
    # 6. Train model
    print("\nStarting training...")
    best_accuracy = trainer.train(
        source_train_loader,
        source_val_loader,
        target_train_loader,
        target_test_loader
    )
    
    print(f"\n✓ Training completed. Best target domain accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
