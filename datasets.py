import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms, datasets
from PIL import Image


class DomainDataset(Dataset):
    """Domain adaptation dataset with flexible structure support."""
    
    def __init__(self, root_dir: str, domain: str, transform=None, return_label: bool = True):
        self.root_dir = root_dir
        self.domain = domain
        self.transform = transform
        self.return_label = return_label
        self.data_list = self._load_data_list()
    
    def _load_data_list(self):
        """Load image paths and labels from directory structure."""
        data_list = []
        domain_dir = os.path.join(self.root_dir, self.domain)
        
        if not os.path.exists(domain_dir):
            raise ValueError(f"Domain directory not found: {domain_dir}")
        
        # Support both Office-31 (class subdirs) and PACS (ImageFolder format)
        class_dirs = [d for d in os.listdir(domain_dir) 
                     if os.path.isdir(os.path.join(domain_dir, d))]
        
        if not class_dirs:
            # Direct image files in domain dir
            for fname in os.listdir(domain_dir):
                if fname.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(domain_dir, fname)
                    label = -1 if not self.return_label else 0
                    data_list.append((img_path, label, 'unknown'))
        else:
            # Class-organized structure
            class_to_idx = {cls: idx for idx, cls in enumerate(class_dirs)}
            for cls_dir in class_dirs:
                cls_path = os.path.join(domain_dir, cls_dir)
                for fname in os.listdir(cls_path):
                    if fname.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(cls_path, fname)
                        label = class_to_idx[cls_dir] if self.return_label else -1
                        data_list.append((img_path, label, cls_dir))
        
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path, label, cls_name = self.data_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, cls_name, img_path


def get_transforms(mode: str = 'source'):
    """Get data transforms for different modes."""
    if mode == 'source':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif mode == 'target_train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
        ])
    else:  # target_test
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])


def get_dataloaders(config):
    """Create dataloaders for single-source domain adaptation."""
    source_transform = get_transforms('source')
    target_train_transform = get_transforms('target_train')
    target_test_transform = get_transforms('target_test')
    
    data_root = os.path.join(config.data_root, config.dataset)
    
    # Source domain with train/val split
    full_source = DomainDataset(
        root_dir=data_root,
        domain=config.source_domain,
        transform=source_transform,
        return_label=True
    )
    
    train_size = int(0.8 * len(full_source))
    val_size = len(full_source) - train_size
    source_train, source_val = random_split(
        full_source, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # Target domain (unlabeled for training, labeled for testing)
    target_train = DomainDataset(
        root_dir=data_root,
        domain=config.target_domain,
        transform=target_train_transform,
        return_label=False
    )
    
    target_test = DomainDataset(
        root_dir=data_root,
        domain=config.target_domain,
        transform=target_test_transform,
        return_label=True
    )
    
    loader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'pin_memory': True
    }
    
    source_train_loader = DataLoader(source_train, shuffle=True, **loader_kwargs)
    source_val_loader = DataLoader(source_val, shuffle=False, **loader_kwargs)
    target_train_loader = DataLoader(target_train, shuffle=True, **loader_kwargs)
    target_test_loader = DataLoader(target_test, shuffle=False, **loader_kwargs)
    
    print(f"Source train: {len(source_train)} | Source val: {len(source_val)}")
    print(f"Target train (unlabeled): {len(target_train)} | Target test: {len(target_test)}")
    
    return source_train_loader, source_val_loader, target_train_loader, target_test_loader


def get_pacs_dataloaders(config):
    """Create dataloaders for PACS dataset with multiple source domains."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Concatenate multiple source domains
    source_datasets = []
    for domain in config.source_domains:
        domain_path = os.path.join(config.data_root, 'PACS', domain)
        source_datasets.append(datasets.ImageFolder(domain_path, transform=train_transform))
    
    full_source = ConcatDataset(source_datasets)
    source_loader = DataLoader(
        full_source,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True
    )
    
    # Target domain
    target_path = os.path.join(config.data_root, 'PACS', config.target_domain)
    target_train = datasets.ImageFolder(target_path, transform=train_transform)
    target_test = datasets.ImageFolder(target_path, transform=test_transform)
    
    target_train_loader = DataLoader(
        target_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True
    )
    
    target_test_loader = DataLoader(
        target_test,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    return source_loader, None, target_train_loader, target_test_loader
