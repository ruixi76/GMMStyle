import os
import torch
import random
import numpy as np
from config import Config
from datasets import get_dataloaders
from model import DomainAdapter
from domain_adapter import GMMStyleDomainAdapter
from pixel_gmm import PixelGaussianMixture

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_target_pixels(target_loader, device, num_batches=1):
    pixel_batches = []
    for batch_idx, (images, _, _, _) in enumerate(target_loader):
        if batch_idx >= max(1, num_batches):
            break
        images = images.to(device)
        pixels = images.permute(0, 2, 3, 1).reshape(-1, 3)
        pixel_batches.append(pixels)
    return torch.cat(pixel_batches, dim=0)


def auto_select_num_gaussians(config, target_loader):
    device = torch.device(config.device)
    target_pixels = collect_target_pixels(
        target_loader,
        device,
        num_batches=config.bic_num_batches,
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
            tol=config.gmm_convergence_threshold,
        )
        bic = gmm.get_bic(target_pixels, max_iters=config.bic_em_iters)
        print(f"[BIC] K={k}, BIC={bic:.2f}")
        if bic < best_bic:
            best_bic = bic
            best_k = k

    config.num_gaussians = best_k
    print(f"[BIC] Selected K={best_k} (min BIC={best_bic:.2f})")

def main():
    # 1. 解析配置
    config = Config().parse()
    set_seed(config.seed)
    
    print(f"Using device: {config.device}")
    if 'cuda' in str(config.device):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. 获取数据加载器
    print("\nLoading datasets...")
    source_train_loader, source_val_loader, target_train_loader, target_test_loader = get_dataloaders(config)

    if getattr(config, 'auto_select_k', False):
        auto_select_num_gaussians(config, target_train_loader)
    
    # 3. 创建模型
    print("\nCreating model...")
    model = DomainAdapter(config)
    
    # 4. 创建训练器
    print("Creating domain adapter...")
    trainer = GMMStyleDomainAdapter(model, config)
    
    # 5. 训练模型
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