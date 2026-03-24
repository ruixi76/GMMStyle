import argparse
import os

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Pixel-level GMM Domain Adaptation')
        
        self.dataset = 'pacs' # 新增数据集标识
        self.num_classes = 7  # PACS 有 7 个类别 (Dog, Elephant, Giraffe, Guitar, Horse, House, Person)
        
        # 将 source_domain 改为列表
        self.source_domains = ['art_painting', 'cartoon', 'sketch'] 
        self.target_domain = 'photo'

        # 数据集配置
        self.parser.add_argument('--source_domain', type=str, default='amazon', 
                               help='Source domain name (e.g., amazon)')
        self.parser.add_argument('--target_domain', type=str, default='dslr', 
                               help='Target domain name (e.g., dslr)')
        self.parser.add_argument('--data_root', type=str, default='/home/amax/paperProject/GMM/data', 
                               help='Root directory of datasets')
        self.parser.add_argument('--dataset', type=str, default='office31', 
                               choices=['office31', 'visda', 'digits', 'pacs'],
                               help='Dataset name')
        
        # 模型配置
        self.parser.add_argument('--backbone', type=str, default='resnet50', 
                               choices=['resnet18', 'resnet50', 'vgg16'],
                               help='Backbone network')
        self.parser.add_argument('--num_classes', type=int, default=31, 
                               help='Number of classes')
        
        # GMM配置
        self.parser.add_argument('--num_gaussians', type=int, default=5, 
                               help='Number of Gaussian components in GMM')
        self.parser.add_argument('--gmm_update_freq', type=int, default=100, 
                               help='Frequency of GMM update (in batches)')
        self.parser.add_argument('--gmm_convergence_threshold', type=float, default=1e-3, 
                               help='Convergence threshold for EM algorithm')
        self.parser.add_argument('--gmm_max_iters', type=int, default=50, 
                               help='Maximum iterations for EM algorithm')
        self.parser.add_argument('--momentum', type=float, default=0.9, 
                               help='Momentum for incremental GMM updates')
        self.parser.add_argument('--covariance_type', type=str, default='diag', 
                               choices=['diag', 'full'],
                               help='Covariance type: diag (faster) or full (more accurate)')
        
        # 训练配置
        self.parser.add_argument('--batch_size', type=int, default=32, 
                               help='Batch size')
        self.parser.add_argument('--num_workers', type=int, default=4, 
                               help='Number of workers for data loading')
        self.parser.add_argument('--epochs', type=int, default=10, 
                               help='Number of training epochs')
        self.parser.add_argument('--lr', type=float, default=0.001, 
                               help='Learning rate')
        self.parser.add_argument('--lr_step_size', type=int, default=15, 
                               help='Learning rate step size')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4, 
                               help='Weight decay')
        self.parser.add_argument('--lambda_div', type=float, default=1.0, 
                               help='Weight for diversity loss (style transferred images)')
        
        # 其他配置
        self.parser.add_argument('--seed', type=int, default=42, 
                               help='Random seed')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                               help='Directory to save checkpoints')
        self.parser.add_argument('--log_dir', type=str, default='./logs', 
                               help='Directory for logs')
        self.parser.add_argument('--device', type=str, default='cuda:1', 
                               help='Device to use (e.g., cuda:1 for GPU 1)')
        self.parser.add_argument('--style_alpha', type=float, default=0.5, 
                       help='Strength of style transfer (0.0 to 1.0)')
    
    def parse(self):
        args = self.parser.parse_args()
        
        # 处理GPU设备映射
        if args.device.startswith('cuda:'):
            gpu_id = int(args.device.split(':')[-1])
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            args.device = 'cuda:0'  # 由于环境变量设置，映射到cuda:0
        
        return args