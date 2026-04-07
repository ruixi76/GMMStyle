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
    """
    把多个 batch 的像素样本合并成一个大样本集合 [num_batches*batch_size*height*width, 3] ，用于后面 BIC 评估。
    params:
    - target_loader: 目标域的 DataLoader。
    - device: 设备（CPU 或 GPU）。
    - num_batches: 从目标域训练集中取多少个 batch 来采样像素。
    returns:
    - 一个形状为 [N, 3] 的张量，包含了 N 个像素的 RGB 值，用于 GMM 拟合和 BIC 评估
    """
    pixel_batches = []
    for batch_idx, (images, _, _, _) in enumerate(target_loader):
        if batch_idx >= max(1, num_batches):
            break
        images = images.to(device)
        pixels = images.permute(0, 2, 3, 1).reshape(-1, 3) # [B，C，H，W] -> [B，H，W，C] -> [B*H*W，C]
        # append 是 Python 列表的“追加”操作。它会把当前这个 batch 的 pixels 张量，放到列表 pixel_batches 的末尾。
        pixel_batches.append(pixels)
    # torch.cat 是“拼接张量。pixel_batches 是一个列表，里面每个元素都是形状类似 [N_i, 3] 的张量。
    # dim=0 表示沿第 0 维（行方向）拼接。
    return torch.cat(pixel_batches, dim=0)


def auto_select_num_gaussians(config, target_loader):
    device = torch.device(config.device)
    # 多个批次的一个大样本集合 [num_batches * batch_size * height * width, 3]
    target_pixels = collect_target_pixels(
        target_loader,
        device,
        num_batches=config.bic_num_batches,
    )

    # config.bic_k_candidates：3，5，7
    print(f"\n[BIC] Evaluating K candidates: {config.bic_k_candidates}") 
    # config.num_gaussians：5 
    best_k = config.num_gaussians # 初始化为默认值，万一 BIC 评估出问题了，就至少有个合理的 K 不会崩溃。
    best_bic = float('inf') # BIC 越小越好，所以初始值设为正无穷。

    for k in config.bic_k_candidates:
        gmm = PixelGaussianMixture(
            num_components=k,
            feature_dim=3,
            device=config.device,
            covariance_type=config.covariance_type, # 'diag'
            max_iters=config.bic_em_iters, # 15
            tol=config.gmm_convergence_threshold, # 1e-3
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

    # getattr(config, 'auto_select_k', False) 是从 config 对象里取 auto_select_k 这个属性。
    # 如果这个属性存在，就返回它的值。
    # 如果不存在，就返回第三个参数里的默认值 False。
    # 单目标域适配任务中，auto_select_k 我们是采取一个批次的数据来进行 BIC 评估的，但若是多目标域适配任务中，
    # 可能会有多个目标域，每个目标域都需要评估一次，所以我们就不限制只取一个批次了，而是每个目标域取一个批次的数据来进行评估。
    if getattr(config, 'auto_select_k', False): # auto_select_k为action='store_true
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