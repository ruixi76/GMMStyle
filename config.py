"""Configuration module for Pixel-level GMM Domain Adaptation."""
import argparse
import os
from typing import List
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = 'office31'
    data_root: str = '/home/amax/paperProject/GMM/data'
    source_domains: List[str] = field(default_factory=lambda: ['art_painting', 'cartoon', 'sketch'])
    target_domain: str = 'photo'
    num_classes: int = 7


@dataclass
class ModelConfig:
    """Model configuration."""
    backbone: str = 'resnet50'
    num_classes: int = 31


@dataclass
class GMMConfig:
    """GMM configuration."""
    num_components: int = 5
    covariance_type: str = 'diag'
    convergence_threshold: float = 1e-3
    max_iters: int = 50
    init_iters: int = 20
    update_freq: int = 100
    momentum: float = 0.9
    ema_tau: float = 0.99


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 15
    learning_rate: float = 0.001
    lr_step_size: int = 15
    weight_decay: float = 1e-4
    seed: int = 42


@dataclass
class StyleTransferConfig:
    """Style transfer configuration."""
    mode: str = 'both'
    lambda_pixel: float = 1.0
    lambda_feature: float = 1.0
    alpha: float = 0.5
    alpha_start: float = 0.3
    alpha_end: float = 0.8
    alpha_warmup_epochs: int = 10


@dataclass
class BICConfig:
    """BIC selection configuration."""
    auto_select: bool = False
    k_candidates: List[int] = field(default_factory=lambda: [3, 5, 7])
    num_batches: int = 1
    em_iters: int = 15


@dataclass
class PathConfig:
    """Path configuration."""
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    device: str = 'cuda:1'


class Config:
    """Main configuration class that aggregates all sub-configurations."""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with all configuration options."""
        parser = argparse.ArgumentParser(description='Pixel-level GMM Domain Adaptation')
        
        # Dataset arguments
        parser.add_argument('--source_domain', type=str, default='amazon',
                          help='Source domain name (e.g., amazon)')
        parser.add_argument('--target_domain', type=str, default='dslr',
                          help='Target domain name (e.g., dslr)')
        parser.add_argument('--data_root', type=str, default='/home/amax/paperProject/GMM/data',
                          help='Root directory of datasets')
        parser.add_argument('--dataset', type=str, default='office31',
                          choices=['office31', 'visda', 'digits', 'pacs'],
                          help='Dataset name')
        
        # Model arguments
        parser.add_argument('--backbone', type=str, default='resnet50',
                          choices=['resnet18', 'resnet50', 'vgg16'],
                          help='Backbone network')
        parser.add_argument('--num_classes', type=int, default=31,
                          help='Number of classes')
        
        # GMM arguments
        parser.add_argument('--num_gaussians', type=int, default=5,
                          help='Number of Gaussian components in GMM')
        parser.add_argument('--gmm_update_freq', type=int, default=100,
                          help='Frequency of GMM update (in batches)')
        parser.add_argument('--gmm_convergence_threshold', type=float, default=1e-3,
                          help='Convergence threshold for EM algorithm')
        parser.add_argument('--gmm_max_iters', type=int, default=50,
                          help='Maximum iterations for EM algorithm')
        parser.add_argument('--gmm_init_iters', type=int, default=20,
                          help='Warmup EM iterations for first target batch')
        parser.add_argument('--momentum', type=float, default=0.9,
                          help='Momentum for incremental GMM updates')
        parser.add_argument('--gmm_ema_tau', type=float, default=0.99,
                          help='EMA factor for online EM updates')
        parser.add_argument('--covariance_type', type=str, default='diag',
                          choices=['diag', 'full'],
                          help='Covariance type: diag (faster) or full (more accurate)')
        
        # Training arguments
        parser.add_argument('--batch_size', type=int, default=32,
                          help='Batch size')
        parser.add_argument('--num_workers', type=int, default=4,
                          help='Number of workers for data loading')
        parser.add_argument('--epochs', type=int, default=15,
                          help='Number of training epochs')
        parser.add_argument('--lr', type=float, default=0.001,
                          help='Learning rate')
        parser.add_argument('--lr_step_size', type=int, default=15,
                          help='Learning rate step size')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                          help='Weight decay')
        parser.add_argument('--lambda_div', type=float, default=1.0,
                          help='Weight for diversity loss (style transferred images)')
        parser.add_argument('--style_mode', type=str, default='both',
                          choices=['pixel', 'feature', 'both'],
                          help='Style transfer mode')
        parser.add_argument('--lambda_pixel', type=float, default=1.0,
                          help='Weight for pixel-level style loss')
        parser.add_argument('--lambda_feature', type=float, default=1.0,
                          help='Weight for feature-level style loss')
        parser.add_argument('--auto_select_k', action='store_true',
                          help='Automatically select K via BIC before training')
        parser.add_argument('--bic_k_candidates', type=str, default='3,5,7',
                          help='Comma-separated candidate K values for BIC selection')
        parser.add_argument('--bic_num_batches', type=int, default=1,
                          help='Number of target batches to use for BIC warmup')
        parser.add_argument('--bic_em_iters', type=int, default=15,
                          help='EM iterations for each BIC candidate')
        
        # Other arguments
        parser.add_argument('--seed', type=int, default=42,
                          help='Random seed')
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                          help='Directory to save checkpoints')
        parser.add_argument('--log_dir', type=str, default='./logs',
                          help='Directory for logs')
        parser.add_argument('--device', type=str, default='cuda:1',
                          help='Device to use (e.g., cuda:1 for GPU 1)')
        parser.add_argument('--style_alpha', type=float, default=0.5,
                          help='Strength of style transfer (0.0 to 1.0)')
        parser.add_argument('--style_alpha_start', type=float, default=0.3,
                          help='Initial style alpha for curriculum schedule')
        parser.add_argument('--style_alpha_end', type=float, default=0.8,
                          help='Final style alpha for curriculum schedule')
        parser.add_argument('--style_alpha_warmup_epochs', type=int, default=10,
                          help='Number of epochs to linearly increase style alpha')
        
        return parser
    
    def parse(self) -> argparse.Namespace:
        """Parse command line arguments and post-process."""
        args = self.parser.parse_args()
        
        # Parse BIC candidates
        args.bic_k_candidates = [
            int(k.strip()) for k in args.bic_k_candidates.split(',') if k.strip()
        ]
        
        # Backward compatibility for lambda_pixel
        if not hasattr(args, 'lambda_pixel'):
            args.lambda_pixel = args.lambda_div
        
        # Configure GPU device mapping
        if args.device.startswith('cuda:'):
            gpu_id = int(args.device.split(':')[-1])
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            args.device = 'cuda:0'
        
        return args
