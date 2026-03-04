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

def get_dataloaders(config):
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