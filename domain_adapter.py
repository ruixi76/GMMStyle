import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pixel_gmm import PixelGaussianMixture
from style_transfer import PixelStyleTransfer, FeatureStyleTransfer
# 导入 transform
from torchvision import transforms

class GMMStyleDomainAdapter:
    """
    基于像素级GMM的域自适应训练器
    工作流程:
    1. 用目标域无标签数据初始化PixelGMM
    2. 每个epoch:
       a. 用源域有标签数据训练分类器
       b. 对源域图像进行像素级风格迁移
       c. 用风格迁移后的图像增强训练
       d. 定期用新目标域数据更新GMM
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        self.conf_threshold = 0.95  # 置信度阈值，建议 0.95
        self.lambda_target = 0.1    # 目标域损失的权重
        self.style_mode = getattr(config, 'style_mode', 'both') # 风格迁移模式
        self.lambda_pixel = getattr(config, 'lambda_pixel', config.lambda_div) # 像素级风格损失权重 lambda_div 默认为1.0
        self.lambda_feature = getattr(config, 'lambda_feature', 1.0) # 特征级风格损失权重
        
        # 移动模型到设备
        self.model = self.model.to(self.device)
        
        # 优化器
        self.optimizer = optim.SGD( # 建议改用 SGD
            self.model.parameters(),
            lr=config.lr, momentum=0.9, weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.lr_step_size if hasattr(config, 'lr_step_size') else 10,
            gamma=0.1
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # GMM模型 (目标域)
        self.target_gmm = PixelGaussianMixture(
            num_components=config.num_gaussians,
            feature_dim=3,  # RGB
            device=config.device,
            covariance_type='diag',  # 推荐使用对角协方差
            max_iters=config.gmm_max_iters, # 50
            tol=config.gmm_convergence_threshold # 1e-3
        )
        
        # 风格迁移模块
        self.style_transfer = PixelStyleTransfer(
            num_components=config.num_gaussians,
            eps=1e-6,
            alpha=config.style_alpha # 从配置读取 0.5
        ).to(self.device)
        self.feature_style_transfer = FeatureStyleTransfer(
            num_components=config.num_gaussians,
            eps=1e-5
        ).to(self.device)
        
        # 目标域统计量 (用于风格迁移)
        self.target_stats = None
        self.target_feature_stats = None
        
        # 训练状态
        self.initialized = False
        self.best_accuracy = 0.0
        
        # 1. 定义 ImageNet 标准化
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def _current_style_alpha(self, batch_idx, total_batches):
        """按单个 epoch 内的 batch 进度线性调度 alpha。"""
        start_alpha = getattr(self.config, 'style_alpha_start', self.config.style_alpha)
        end_alpha = getattr(self.config, 'style_alpha_end', self.config.style_alpha)
        ratio = float(batch_idx + 1) / float(max(1, total_batches)) # batch_idx 从 0 开始，所以加 1；total_batches 是总批次数，确保不除以零。
        ratio = min(max(ratio, 0.0), 1.0) # 确保 ratio 在 [0,1] 范围内
        return start_alpha + (end_alpha - start_alpha) * ratio
    
    def _compute_source_statistics(self, source_images, responsibilities):
        """
        计算源域分量级统计量（软分配）
        """
        B, C, H, W = source_images.shape
        
        # 转换为像素形式
        pixels = source_images.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        flat_responsibilities = responsibilities.reshape(-1, self.config.num_gaussians)  # [B*H*W, K]

        source_stats = self.target_gmm.get_soft_component_statistics(pixels, flat_responsibilities)

        return source_stats

    def _compute_target_statistics(self, target_images):
        """用冻结目标 GMM 对目标批次做 E 步，得到软分配统计量。"""
        B, C, H, W = target_images.shape
        target_pixels = target_images.permute(0, 2, 3, 1).reshape(-1, C)
        _, target_probs = self.target_gmm.predict(target_images)  # [B, H, W, K]，得到的是E步的软分配概率
        target_responsibilities = target_probs.reshape(-1, self.config.num_gaussians)
         # 其中和 m 步的计算"diag"型协方差矩阵的方法一样，为什么不直接用EM算法的 M 步来计算呢？因为我们在训练过程中是在线更新 GMM 的参数的，
         # 所以每个批次的目标域数据都会有一个新的软分配统计量，这个统计量是根据当前 GMM 参数对当前批次数据做 E 步得到的。
         # 我们需要这个统计量来进行风格迁移，而不是直接用 EM 算法的 M 步计算出来的参数，因为 M 步计算出来的是全局的 GMM 参数，而我们需要的是针对当前批次数据的统计量，这样才能更准确地进行风格迁移。
        target_stats = self.target_gmm.get_soft_component_statistics(target_pixels, target_responsibilities)

        return target_stats

    def _compute_image_level_responsibilities(self, pixel_responsibilities):
        """从 [B,H,W,K] 聚合为每张图的 [B,K] 软分配。"""
        return torch.mean(pixel_responsibilities, dim=(1, 2))

    def _compute_soft_feature_statistics(self, features, responsibilities):
        """按软分配统计特征均值与满秩协方差。"""
        batch_size, feat_dim = features.shape
        k = self.config.num_gaussians
        means = torch.zeros(k, feat_dim, device=self.device, dtype=features.dtype)
        covariances = torch.zeros(k, feat_dim, feat_dim, device=self.device, dtype=features.dtype)
        counts = torch.zeros(k, device=self.device, dtype=features.dtype)

        eye = torch.eye(feat_dim, device=self.device, dtype=features.dtype)
        for comp in range(k):
            gamma = responsibilities[:, comp]
            nk = torch.sum(gamma) + 1e-8
            counts[comp] = nk

            mean = torch.sum(gamma.unsqueeze(1) * features, dim=0) / nk
            centered = features - mean.unsqueeze(0)
            weighted = gamma.unsqueeze(1) * centered
            cov = (weighted.T @ centered) / nk

            means[comp] = mean
            covariances[comp] = cov + 1e-5 * eye

        return {
            'means': means,
            'covariances': covariances,
            'counts': counts,
        }

    def _compute_target_feature_statistics(self, target_images, target_probs):
        with torch.no_grad():
            target_inputs = self.normalize(target_images)
            target_features = self.model.extract_features(target_inputs)
            target_image_resp = self._compute_image_level_responsibilities(target_probs)
            return self._compute_soft_feature_statistics(target_features, target_image_resp)
    
    def train_epoch(self, source_loader, target_loader, epoch):
        """
        训练一个epoch
        """
        # if not self.initialized:
        #     raise RuntimeError("GMM not initialized. Call initialize_with_target_domain first.")
        
        self.model.train() # 设置为训练模式
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # 目标域迭代器 (用于定期更新GMM)
        target_iter = iter(target_loader) # target_train_loader没有标签
        
        # 训练进度
        # 如果 source_loader 原本产出的是 (图片，标签，...)，加上 enumerate 后，产出的数据变成了 (索引，(图片，标签，...))。
        # total=len(source_loader):告诉 tqdm 总共需要迭代多少次。
        # desc 是 "description" 的缩写，设置进度条左侧的描述文字。
        total_batches = len(source_loader)
        pbar = tqdm(enumerate(source_loader), total=total_batches, desc=f'Epoch {epoch+1}')
        
        # 从 enumerate(source_loader) 中取出一个元组，结构是：(batch_idx, batch_data)。
        for batch_idx, (source_images, labels, _, _) in pbar:
            # 计算当前 batch 的 alpha（风格迁移强度），根据 epoch 内的 batch 进度线性调度。
            style_alpha = self._current_style_alpha(batch_idx, total_batches) # 计算当前 batch 的 alpha
            source_images = source_images.to(self.device)
            labels = labels.to(self.device)
            B, C, H, W = source_images.shape

            try:
                # 获取一个 batch 的目标域数据
                target_images, _, _, _ = next(target_iter) # target_train_loader没有标签
            except StopIteration:
                # 如果遍历完了，重新开始
                target_iter = iter(target_loader)
                target_images, _, _, _ = next(target_iter)

            target_images = target_images.to(self.device)

            if not self.target_gmm.initialized:
                print("\n[Train] Lazy initializing GMM with the first target batch...")
                # 将图像展平为像素 B,C,H,W->B,H,W,C->B*H*W,C
                target_pixels = target_images.permute(0, 2, 3, 1).reshape(-1, 3)

                # 首批次进行较充分的 warmup EM
                self.target_gmm.fit(
                    target_pixels,
                    max_iters=getattr(self.config, 'gmm_init_iters', 20), # gmm_init_iters:20
                    return_details=True
                )

                self.target_stats = self._compute_target_statistics(target_images)
                _, target_probs = self.target_gmm.predict(target_images)
                self.target_feature_stats = self._compute_target_feature_statistics(target_images, target_probs)
                self.initialized = True
                print("[Train] GMM initialized successfully.")

            # 1. 像素级分配到GMM分量
            # assignments 是一个维度为[B,H,W]的张量，表示每个像素被分配到的分量索引（从 0 到 num_components-1）。
            _, source_probs = self.target_gmm.predict(source_images)  # [B, H, W, K]
            
            # 清空上一个 Batch 残留的梯度（像计算器归零）。
            self.optimizer.zero_grad()

            source_inputs = self.normalize(source_images)
            target_inputs = self.normalize(target_images)

            # 原始源域图像
            source_features = self.model.extract_features(source_inputs)
            logits_src = self.model.classify_features(source_features)
            loss_src = self.criterion(logits_src, labels)

            loss_pixel = torch.tensor(0.0, device=self.device)
            loss_feature = torch.tensor(0.0, device=self.device)

            if self.style_mode in ('pixel', 'both'):
                source_stats = self._compute_source_statistics(source_images, source_probs)
                styled_images = self.style_transfer(
                    source_images,
                    source_probs,
                    source_stats,
                    self.target_stats,
                    alpha=style_alpha
                )
                styled_inputs = self.normalize(styled_images)
                logits_pixel, _ = self.model(styled_inputs)
                loss_pixel = self.criterion(logits_pixel, labels)

            if self.style_mode in ('feature', 'both') and self.target_feature_stats is not None:
                source_img_resp = self._compute_image_level_responsibilities(source_probs)
                source_feature_stats = self._compute_soft_feature_statistics(
                    source_features.detach(),
                    source_img_resp.detach()
                )
                stylized_features = self.feature_style_transfer(
                    source_features,
                    source_img_resp,
                    source_feature_stats,
                    self.target_feature_stats
                )
                logits_feature = self.model.classify_features(stylized_features)
                loss_feature = self.criterion(logits_feature, labels)
            
            # 目标域伪标签训练 (Pseudo-Labeling)
            # 1. 计算目标域 logits
            logits_tgt, _ = self.model(target_inputs)
            
            # 2. 计算概率
            probs_tgt = torch.softmax(logits_tgt, dim=1)
            
            # 3. 获取最大概率和对应的预测类别
            max_probs, tgt_preds = torch.max(probs_tgt, dim=1)
            
            # 4. 生成掩码：只选择置信度 > 0.95 的样本
            mask = max_probs.ge(self.conf_threshold).float()
            
            # 5. 计算损失 (只计算高置信度样本)
            # reduction='none' 是为了保留每个样本的损失，以便乘以 mask
            # loss_target_raw = nn.CrossEntropyLoss(reduction='none')(logits_target, target_preds)
            # loss_target = (loss_target_raw * mask).mean()
            loss_tgt = (nn.CrossEntropyLoss(reduction='none')(logits_tgt, tgt_preds) * mask).mean()

            loss = loss_src + self.lambda_target * loss_tgt
            if self.style_mode in ('pixel', 'both'):
                loss = loss + self.lambda_pixel * loss_pixel
            if self.style_mode in ('feature', 'both'):
                loss = loss + self.lambda_feature * loss_feature

            # 5. 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 6. 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits_src.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # 7. 更新进度条
            pbar.set_postfix({
                'Loss': f"{total_loss/(batch_idx+1):.4f}",
                'Acc': f"{100.*total_correct/total_samples:.2f}%",
                'a': f"{style_alpha:.2f}",
                'Lp': f"{loss_pixel.item():.3f}",
                'Lf': f"{loss_feature.item():.3f}"
            })
            
            # 8. 定期更新GMM (每100个批次)
            if (batch_idx + 1) % self.config.gmm_update_freq == 0:
                target_pixels = target_images.permute(0, 2, 3, 1).reshape(-1, 3)
                self.target_gmm.online_em_update(target_pixels, tau=self.config.gmm_ema_tau)
                self.target_stats = self._compute_target_statistics(target_images)
                _, target_probs = self.target_gmm.predict(target_images)
                self.target_feature_stats = self._compute_target_feature_statistics(target_images, target_probs)

        return total_loss / len(source_loader), 0.0 # 返回 acc
    
    # 以下是你在跑 MixStyle baseline 时，替代原先 train_epoch 的逻辑片段

    def train_epoch_mixstyle(self, source_loader, target_loader, epoch):
        self.model.train()
        target_iter = iter(target_loader)
        
        # 训练进度
        pbar = tqdm(enumerate(source_loader), total=len(source_loader), desc=f'Epoch {epoch+1}')

        for batch_idx, (source_images, labels, _, _) in enumerate(source_loader):
            source_images = source_images.to(self.device)
            labels = labels.to(self.device)
            
            try:
                target_images, _, _, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_images, _, _, _ = next(target_iter)
            target_images = target_images.to(self.device)

            # 加上 ImageNet 标准化 (因为没有你的像素级操作，直接正常输入即可)
            source_images = self.normalize(source_images)
            target_images = self.normalize(target_images)

            # 核心：将 Source 和 Target 拼接在一起，送入含有 MixStyle 的网络
            # 这样网络内部的 MixStyle 就能把 Source 和 Target 的底层特征统计量进行混合
            combined_images = torch.cat([source_images, target_images], dim=0)
            
            self.optimizer.zero_grad()
            
            # 前向传播 (此时 MixStyle 会在特征层自动发生)
            logits_combined, _ = self.model(combined_images)
            
            # 拆分预测结果
            batch_src = source_images.size(0)
            logits_src = logits_combined[:batch_src]
            logits_tgt = logits_combined[batch_src:]
            
            # 1. 源域分类损失
            loss_src = self.criterion(logits_src, labels)
            
            # 2. 目标域伪标签损失 (为了与你的方法公平对比，保留你的伪标签学习策略)
            probs_tgt = torch.softmax(logits_tgt, dim=1)
            max_probs, tgt_preds = torch.max(probs_tgt, dim=1)
            mask = max_probs.ge(self.conf_threshold).float()
            loss_tgt = (nn.CrossEntropyLoss(reduction='none')(logits_tgt, tgt_preds) * mask).mean()
            
            # 只需要这两个损失即可，没有 loss_style 了
            loss = loss_src + self.lambda_target * loss_tgt
            
            loss.backward()
            self.optimizer.step()

            # 7. 更新进度条

    
    def validate(self, loader, domain_name='target'):
        """
        在指定域上验证模型
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels, _, _ in tqdm(loader, desc=f'Validating {domain_name}'):
                images = images.to(self.device)
                images = self.normalize(images)
                labels = labels.to(self.device)
                
                outputs, _ = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = 100. * total_correct / total_samples
        print(f"{domain_name.capitalize()} domain accuracy: {accuracy:.2f}%")
        
        return accuracy
    
    def train(self, source_train_loader, source_val_loader, 
             target_train_loader, target_test_loader):
        """
        完整训练流程
        """
        # 2. 训练循环
        for epoch in range(self.config.epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"{'='*50}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(
                source_train_loader, 
                target_train_loader, 
                epoch
            )
            
            # 源域验证 (仅用于监控)
            source_acc = self.validate(source_val_loader, 'source')
            
            # 目标域测试 (关键指标)
            if (epoch + 1) % 5 == 0 or epoch == self.config.epochs - 1:
                target_acc = self.validate(target_test_loader, 'target')
                
                # 保存最佳模型
                if target_acc > self.best_accuracy:
                    self.best_accuracy = target_acc
                    self._save_checkpoint(epoch, target_acc, is_best=True)
                    print(f"✓ New best model saved: {target_acc:.2f}%")
            
            # 学习率调度
            self.scheduler.step()
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, self.best_accuracy)
        
        print(f"\n{'='*50}")
        print(f"Training completed. Best target accuracy: {self.best_accuracy:.2f}%")
        print(f"{'='*50}")
        
        return self.best_accuracy
    
    def _save_checkpoint(self, epoch, accuracy, is_best=False):
        """保存模型检查点"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'best_accuracy': self.best_accuracy,
            'gmm_means': self.target_gmm.means,
            'gmm_covariances': self.target_gmm.covariances,
            'gmm_weights': self.target_gmm.weights,
            'target_stats': self.target_stats
        }
        
        # 保存最新模型
        filename = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(state, filename)
        
        # 保存最佳模型
        if is_best:
            best_filename = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_filename)