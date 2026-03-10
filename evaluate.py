import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from config import Config
from datasets import get_dataloaders
from model import DomainAdapter
from pixel_gmm import PixelGaussianMixture
from style_transfer import PixelStyleTransfer
# 引入 transform
from torchvision import transforms

def load_checkpoint(checkpoint_path, config, device):
    """加载训练好的模型和GMM参数"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型
    model = DomainAdapter(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 重建GMM
    gmm = PixelGaussianMixture(
        num_components=config.num_gaussians,
        feature_dim=3,
        device=device,
        covariance_type=config.covariance_type if hasattr(config, 'covariance_type') else 'diag',
        max_iters=config.gmm_max_iters if hasattr(config, 'gmm_max_iters') else 50,
        tol=config.gmm_convergence_threshold if hasattr(config, 'gmm_convergence_threshold') else 1e-3
    )
    
    # 加载GMM参数
    if 'gmm_means' in checkpoint:
        gmm.means = checkpoint['gmm_means'].to(device)
        gmm.covariances = checkpoint['gmm_covariances'].to(device)
        gmm.weights = checkpoint['gmm_weights'].to(device)
        gmm.initialized = True
        print(f"GMM parameters loaded: {config.num_gaussians} components")
    else:
        print("Warning: GMM parameters not found in checkpoint")
    
    # 加载目标域统计量
    target_stats = None
    if 'target_stats' in checkpoint:
        target_stats = {
            'means': checkpoint['target_stats']['means'].to(device),
            'stds': checkpoint['target_stats']['stds'].to(device),
            'counts': checkpoint['target_stats']['counts'].to(device)
        }
        print("Target domain statistics loaded")
    
    return model, gmm, target_stats, checkpoint.get('accuracy', 0.0)

def evaluate(model, gmm, target_stats, dataloader, device, config, save_dir='./eval_results'):
    """在目标域测试集上评估模型"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []
    all_assignments = []
    
    correct = 0
    total = 0
    
    # 风格迁移模块
    style_transfer = PixelStyleTransfer(
        num_components=config.num_gaussians,
        eps=1e-6
    ).to(device)

    # 添加标准化层
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ).to(device)
    
    with torch.no_grad():
        for batch_idx, (images, labels, _, _) in enumerate(tqdm(dataloader, desc='Evaluating')):
            images = images.to(device)
            labels = labels.to(device)
            
            # 1. 像素级分配
            assignments, _ = gmm.predict(images)
            
            # 2. 风格迁移（可选：评估风格迁移后的性能）
            if target_stats is not None:
                # 计算源域统计量
                B, C, H, W = images.shape
                pixels = images.permute(0, 2, 3, 1).reshape(-1, C)
                flat_assignments = assignments.reshape(-1)
                
                source_stats = {
                    'means': torch.zeros(config.num_gaussians, C, device=device),
                    'stds': torch.zeros(config.num_gaussians, C, device=device)
                }
                
                for k in range(config.num_gaussians):
                    mask = (flat_assignments == k)
                    if torch.sum(mask) > 0:
                        pixels_k = pixels[mask]
                        source_stats['means'][k] = torch.mean(pixels_k, dim=0)
                        source_stats['stds'][k] = torch.std(pixels_k, dim=0) + 1e-8
                
                # 风格迁移
                styled_images = style_transfer(images, assignments, source_stats, target_stats)
                inputs = styled_images
            else:
                inputs = images
            
            # 【修复1】加上标准化
            inputs = normalize(inputs)
            # 3. 预测
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 4. 统计
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 保存部分样本用于可视化
            if batch_idx < 5:  # 仅保存前5个批次
                all_images.append(images.cpu())
                all_assignments.append(assignments.cpu())
    
    # 计算准确率
    accuracy = 100.0 * correct / total
    print(f"\nTarget Domain Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # 生成分类报告
    class_names = []
    domain_dir = os.path.join(config.data_root, config.dataset, config.target_domain)
    if os.path.exists(domain_dir):
        class_names = sorted([d for d in os.listdir(domain_dir) 
                            if os.path.isdir(os.path.join(domain_dir, d))])
    
    if len(class_names) > 0:
        report = classification_report(all_labels, all_preds, 
                                     target_names=class_names, 
                                     digits=4, 
                                     zero_division=0)
        print("\nClassification Report:")
        print(report)
        
        # 保存报告
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 可视化样本预测
    if len(all_images) > 0:
        visualize_predictions(
            all_images[:3], 
            all_assignments[:3], 
            all_preds[:3*all_images[0].shape[0]], 
            all_labels[:3*all_images[0].shape[0]],
            save_path=os.path.join(save_dir, 'prediction_samples.png')
        )
    
    return accuracy, all_preds, all_labels, cm

def visualize_predictions(images_list, assignments_list, preds, labels, save_path='prediction_samples.png'):
    """可视化预测结果"""
    num_samples = min(12, len(preds))
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    sample_idx = 0
    for batch_idx in range(len(images_list)):
        images = images_list[batch_idx]
        assignments = assignments_list[batch_idx]
        
        for i in range(images.shape[0]):
            if sample_idx >= num_samples or sample_idx >= len(preds):
                break
            
            # 原始图像
            img = images[i].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            # 像素分配可视化
            assign_img = assignments[i].numpy()
            
            # 创建分配覆盖图
            overlay = img.copy()
            for k in range(np.max(assign_img) + 1):
                mask = (assign_img == k)
                if np.any(mask):
                    # 为不同分量使用不同颜色
                    color = plt.cm.tab10(k % 10)[:3]
                    overlay[mask] = overlay[mask] * 0.6 + np.array(color) * 0.4
            
            # 绘制
            axes[sample_idx].imshow(overlay)
            title = f"Pred: {preds[sample_idx]}\nTrue: {labels[sample_idx]}"
            if preds[sample_idx] == labels[sample_idx]:
                title += " ✓"
                axes[sample_idx].set_title(title, color='green', fontsize=10)
            else:
                title += " ✗"
                axes[sample_idx].set_title(title, color='red', fontsize=10)
            axes[sample_idx].axis('off')
            
            sample_idx += 1
            if sample_idx >= num_samples:
                break
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Prediction samples saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate GMM Domain Adaptation Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--source_domain', type=str, default='amazon', help='Source domain name')
    parser.add_argument('--target_domain', type=str, default='dslr', help='Target domain name')
    parser.add_argument('--dataset', type=str, default='office31', choices=['office31', 'visda', 'digits', 'pacs'], help='Dataset name')
    parser.add_argument('--num_classes', type=int, default=31, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory of datasets')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet18', 'resnet50', 'vgg16'], help='Backbone network')
    parser.add_argument('--num_gaussians', type=int, default=5, help='Number of Gaussian components')
    parser.add_argument('--eval_with_style', action='store_true', help='Evaluate with style transfer')
    parser.add_argument('--save_dir', type=str, default='./eval_results', help='Directory to save evaluation results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    # 新增缺失的GMM相关参数
    parser.add_argument('--gmm_max_iters', type=int, default=50, help='Maximum iterations for EM algorithm')
    parser.add_argument('--gmm_convergence_threshold', type=float, default=1e-3, help='Convergence threshold for EM algorithm')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for GMM parameter updates')
    parser.add_argument('--covariance_type', type=str, default='diag', choices=['diag', 'full'], help='Covariance type for GMM')
    # 新增训练相关参数
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--lambda_div', type=float, default=1.0, help='Weight for diversity loss')
    parser.add_argument('--lr_step_size', type=int, default=15, help='Learning rate step size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for logs')
    
    args = parser.parse_args()
    
    # 处理GPU设备
    if args.device.startswith('cuda:'):
        gpu_id = int(args.device.split(':')[-1])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        args.device = 'cuda:0'
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建配置对象 - 包含所有必要属性
    class EvalConfig:
        pass
    
    config = EvalConfig()
    # 基本配置
    config.source_domain = args.source_domain
    config.target_domain = args.target_domain
    config.dataset = args.dataset
    config.num_classes = args.num_classes
    config.batch_size = args.batch_size
    config.device = device
    config.data_root = args.data_root
    config.num_workers = args.num_workers
    config.backbone = args.backbone
    config.num_gaussians = args.num_gaussians
    config.seed = args.seed
    
    # GMM相关配置（修复缺失属性）
    config.gmm_max_iters = args.gmm_max_iters
    config.gmm_convergence_threshold = args.gmm_convergence_threshold
    config.momentum = args.momentum
    config.covariance_type = args.covariance_type
    
    # 训练相关配置（避免后续错误）
    config.lr = args.lr
    config.weight_decay = args.weight_decay
    config.lambda_div = args.lambda_div
    config.lr_step_size = args.lr_step_size
    config.epochs = args.epochs
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir
    
    # 获取数据加载器
    print("\nLoading target domain test data...")
    _, _, _, target_test_loader = get_dataloaders(config)
    
    # 加载模型和GMM
    model, gmm, target_stats, best_acc = load_checkpoint(args.checkpoint, config, device)
    print(f"Loaded model with previous accuracy: {best_acc:.2f}%")
    
    # 评估
    print("\nEvaluating model on target domain...")
    accuracy, preds, labels, cm = evaluate(
        model, 
        gmm, 
        target_stats, 
        target_test_loader, 
        device, 
        config,
        save_dir=args.save_dir
    )
    
    print(f"\n{'='*60}")
    print(f"Evaluation completed. Final accuracy: {accuracy:.2f}%")
    print(f"Results saved to {args.save_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()