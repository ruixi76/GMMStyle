import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import random

class DomainDataset(Dataset):
    def __init__(self, root_dir, domain, transform=None, return_label=True):
        self.root_dir = root_dir
        self.domain = domain
        self.transform = transform
        self.return_label = return_label
        
        # 加载文件列表
        self.data_list = self._load_data_list()
    
    def _load_data_list(self):
        data_list = []
        domain_dir = os.path.join(self.root_dir, self.domain)
        
        if not os.path.exists(domain_dir):
            raise ValueError(f"Domain directory {domain_dir} does not exist")
        
        # Office-31数据集结构: 每个类别一个文件夹
        class_dirs = [d for d in os.listdir(domain_dir) 
                     if os.path.isdir(os.path.join(domain_dir, d))]
        class_to_idx = {class_dir: idx for idx, class_dir in enumerate(class_dirs)}
        
        for class_dir in class_dirs:
            class_path = os.path.join(domain_dir, class_dir)
            image_files = [f for f in os.listdir(class_path) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                label = class_to_idx[class_dir] if self.return_label else -1
                data_list.append((img_path, label, class_dir))
        
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path, label, class_name = self.data_list[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, class_name, img_path

def  get_dataloaders(config):
    """获取正确的域自适应数据加载器"""
    
    # 源域转换 (不使用ImageNet标准化，保持像素值在[0,1])
    source_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 输出范围[0,1]
    ])
    
    # 目标域转换
    target_train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
    ])
    
    target_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # 创建源域数据集 (Amazon)
    full_source_dataset = DomainDataset(
        root_dir=os.path.join(config.data_root, config.dataset),
        domain=config.source_domain,
        transform=source_transform,
        return_label=True
    )
    
    # 分割源域: 80%训练, 20%验证
    train_size = int(0.8 * len(full_source_dataset))
    val_size = len(full_source_dataset) - train_size
    source_train_dataset, source_val_dataset = random_split(
        full_source_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # 目标域训练数据集 (无标签)
    target_train_dataset = DomainDataset(
        root_dir=os.path.join(config.data_root, config.dataset),
        domain=config.target_domain,
        transform=target_train_transform,
        return_label=False  # 无标签
    )
    
    # 目标域测试数据集 (有标签，仅用于评估)
    target_test_dataset = DomainDataset(
        root_dir=os.path.join(config.data_root, config.dataset),
        domain=config.target_domain,
        transform=target_test_transform,
        return_label=True
    )
    
    # 创建数据加载器
    # shuffle=True：每个 epoch 开始时都会打乱数据顺序，避免模型按固定顺序学习，训练更稳定。
    # num_workers=config.num_workers：用于数据加载的并行子进程数量。
    #                               0 表示主进程加载（最稳但可能慢）
    #                               >0 表示并行预取（通常更快，Linux 下常用 2/4/8 试出来）
    # pin_memory=True：把 batch 放到锁页内存（pinned memory）里，GPU 训练时从 CPU 拷到 GPU 通常更快（配合 non_blocking=True 更明显）。
    #                   如果只用 CPU，收益很小，可设 False。
    source_train_loader = DataLoader(
        source_train_dataset, batch_size=config.batch_size, 
        shuffle=True, num_workers=config.num_workers, pin_memory=True
    )
    
    source_val_loader = DataLoader(
        source_val_dataset, batch_size=config.batch_size, 
        shuffle=False, num_workers=config.num_workers, pin_memory=True
    )
    
    target_train_loader = DataLoader(
        target_train_dataset, batch_size=config.batch_size, 
        shuffle=True, num_workers=config.num_workers, pin_memory=True
    )
    
    target_test_loader = DataLoader(
        target_test_dataset, batch_size=config.batch_size, 
        shuffle=False, num_workers=config.num_workers, pin_memory=True
    )
    
    print(f"Source train: {len(source_train_loader.dataset)} images")
    print(f"Source val: {len(source_val_loader.dataset)} images")
    print(f"Target train (unlabeled): {len(target_train_loader.dataset)} images")
    print(f"Target test: {len(target_test_loader.dataset)} images")
    
    return source_train_loader, source_val_loader, target_train_loader, target_test_loader

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

def get_pacs_dataloaders(config):
    # 1. 定义数据增强 (参考 MixStyle 的标准设置)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # 注意：这里依然是 ToTensor，标准化在 evaluate 或网络内部做
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 2. 加载多个源域并合并 (MSDA 核心)
    source_datasets = []
    for domain in config.source_domains:
        domain_path = os.path.join(config.data_root, 'PACS', domain)
        ds = datasets.ImageFolder(root=domain_path, transform=train_transform)
        source_datasets.append(ds)
    
    # 将多个源域数据集拼接为一个巨大的训练集
    full_source_dataset = ConcatDataset(source_datasets)
    
    source_loader = DataLoader(
        full_source_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        drop_last=True
    )

    # 3. 加载目标域训练集 (用于 GMM 聚类和伪标签学习)
    target_train_path = os.path.join(config.data_root, 'PACS', config.target_domain)
    target_train_dataset = datasets.ImageFolder(root=target_train_path, transform=train_transform)
    target_train_loader = DataLoader(
        target_train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        drop_last=True
    )

    # 4. 加载目标域测试集 (用于 evaluate)
    target_test_dataset = datasets.ImageFolder(root=target_train_path, transform=test_transform)
    target_test_loader = DataLoader(
        target_test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )

    return source_loader, None, target_train_loader, target_test_loader