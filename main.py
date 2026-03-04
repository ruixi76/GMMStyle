import os
import torch
import random
import numpy as np
from config import Config
from datasets import get_dataloaders
from model import DomainAdapter
from domain_adapter import GMMStyleDomainAdapter

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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