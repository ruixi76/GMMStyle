import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from tqdm import tqdm
# 引入 sklearn 进行 K-Means 初始化
from sklearn.cluster import KMeans

class PixelGaussianMixture:
    """
    像素级高斯混合模型 - 纯PyTorch实现，避免CPU/GPU切换
    理论基础: p(x) = Σ_k π_k * N(x|μ_k, Σ_k)
    """
    def __init__(self, num_components=5, feature_dim=3, device='cuda:0', 
                 covariance_type='diag', max_iters=100, tol=1e-4):
        """
        Args:
            num_components: 高斯分量数量k
            feature_dim: 特征维度 (RGB图像为3)
            device: 计算设备
            covariance_type: 'full' (完整协方差) 或 'diag' (对角协方差，推荐)
            max_iters: EM算法最大迭代次数
            tol: 收敛阈值
        """
        self.num_components = num_components
        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.covariance_type = covariance_type
        self.max_iters = max_iters
        self.tol = tol
        
        # GMM参数 (全部在GPU上)
        self.means = None          # [k, C] - 每个分量的均值向量
        self.covariances = None    # [k, C, C] - 每个分量的协方差矩阵
        self.weights = None        # [k] - 混合权重
        
        # 初始化为None，等待数据初始化
        self.initialized = False
    
    def _initialize_parameters(self, pixels):
        """
        使用 K-Means 初始化 GMM 参数 (更稳定)
        Args:
            pixels: 像素数据 [N, C]
        """
        print(f"[PixelGMM] Initializing parameters using K-Means (k={self.num_components})...")
        N = pixels.shape[0]
        
        # 1. 转为 CPU numpy 数组以使用 sklearn
        pixels_cpu = pixels.detach().cpu().numpy()
        
        # 2. 运行 K-Means
        # n_init=10 意味着运行10次取最优，防止局部最优
        kmeans = KMeans(n_clusters=self.num_components, n_init=10, random_state=42)
        labels = kmeans.fit_predict(pixels_cpu)
        centers = kmeans.cluster_centers_
        
        # 3. 初始化均值 (Means)
        self.means = torch.tensor(centers, device=self.device, dtype=pixels.dtype)
        
        # 4. 初始化权重 (Weights)
        counts = np.bincount(labels, minlength=self.num_components)
        weights = counts / counts.sum()
        self.weights = torch.tensor(weights, device=self.device, dtype=pixels.dtype)
        
        # 5. 初始化协方差 (Covariances)
        # 计算每个簇的经验协方差
        if self.covariance_type == 'diag':
            self.covariances = torch.zeros(self.num_components, self.feature_dim, 
                                         self.feature_dim, device=self.device)
        else:
            self.covariances = torch.zeros(self.num_components, self.feature_dim, 
                                         self.feature_dim, device=self.device)

        # 将 labels 转回 GPU 以便计算
        labels_torch = torch.tensor(labels, device=self.device)

        for k in range(self.num_components):
            mask = (labels_torch == k)
            if mask.sum() > 1: # 至少有两个点才能算方差
                cluster_pixels = pixels[mask]
                # 计算方差/协方差
                if self.covariance_type == 'diag':
                    var = torch.var(cluster_pixels, dim=0, unbiased=True)
                    self.covariances[k] = torch.diag(var.clamp(min=1e-4))
                else:
                    centered = cluster_pixels - self.means[k]
                    cov = torch.matmul(centered.T, centered) / (mask.sum() - 1)
                    self.covariances[k] = cov + 1e-6 * torch.eye(self.feature_dim, device=self.device)
            else:
                # 兜底：如果簇内样本太少，使用全局方差或单位阵
                self.covariances[k] = torch.eye(self.feature_dim, device=self.device) * 0.01

        self.initialized = True
        print(f"[PixelGMM] Initialization complete on {self.device}")
    
    def e_step(self, pixels):
        """
        E步: 计算每个像素属于各高斯分量的后验概率
        Args:
            pixels: 像素数据 [N, C]
        Returns:
            responsibilities: 责任矩阵 [N, k]
            log_likelihood: 对数似然 (标量)
        """
        N = pixels.shape[0]
        log_probs = torch.zeros(N, self.num_components, device=self.device)
        
        # 计算每个像素到各高斯分量的对数概率
        for k in range(self.num_components):
            # 创建多变量正态分布
            # 注意: PyTorch的MultivariateNormal需要协方差矩阵可逆
            try:
                dist = MultivariateNormal(self.means[k], self.covariances[k] + 1e-6 * torch.eye(self.feature_dim, device=self.device))
                log_prob = dist.log_prob(pixels)  # [N, C] - 每个像素的对数概率
            except:
                # 协方差矩阵奇异时的回退方案
                diff = pixels - self.means[k]
                if self.covariance_type == 'diag':
                    inv_cov = 1.0 / (torch.diagonal(self.covariances[k]) + 1e-8)
                    log_det = torch.sum(torch.log(torch.diagonal(self.covariances[k]) + 1e-8))
                    mahalanobis = torch.sum(diff * diff * inv_cov, dim=1)
                else:
                    # 使用Cholesky分解计算马氏距离
                    try:
                        L = torch.linalg.cholesky(self.covariances[k] + 1e-6 * torch.eye(self.feature_dim, device=self.device))
                        y = torch.linalg.solve_triangular(L, diff.T, upper=False)
                        mahalanobis = torch.sum(y.T ** 2, dim=1)
                        log_det = 2 * torch.sum(torch.log(torch.diagonal(L)))
                    except:
                        # 最后的回退: 使用欧氏距离
                        mahalanobis = torch.sum(diff ** 2, dim=1)
                        log_det = self.feature_dim * np.log(2 * np.pi)
                
                log_prob = -0.5 * (self.feature_dim * np.log(2 * np.pi) + log_det + mahalanobis)
            
            # 加上混合权重的对数
            log_probs[:, k] = log_prob + torch.log(self.weights[k] + 1e-10)
        
        # 数值稳定的归一化 (log-sum-exp trick)
        max_log_prob = torch.max(log_probs, dim=1, keepdim=True)[0]
        probs = torch.exp(log_probs - max_log_prob)
        responsibilities = probs / (torch.sum(probs, dim=1, keepdim=True) + 1e-10)
        
        # 计算总对数似然
        log_likelihood = torch.sum(max_log_prob.squeeze() + 
                                  torch.log(torch.sum(probs, dim=1) + 1e-10))
        
        return responsibilities, log_likelihood.item()
    
    def m_step(self, pixels, responsibilities):
        """
        M步: 基于责任矩阵更新GMM参数
        Args:
            pixels: 像素数据 [N, C]
            responsibilities: 责任矩阵 [N, k]
        """
        N = pixels.shape[0]
        new_means = torch.zeros_like(self.means)
        new_covariances = torch.zeros_like(self.covariances)
        new_weights = torch.zeros_like(self.weights)
        
        for k in range(self.num_components):
            resp_k = responsibilities[:, k]  # [N]
            Nk = torch.sum(resp_k) + 1e-10
            
            # 更新均值: μ_k = Σ_n γ_k(x_n) * x_n / Σ_n γ_k(x_n)
            new_means[k] = torch.sum(resp_k.unsqueeze(1) * pixels, dim=0) / Nk
            
            # 更新协方差矩阵: Σ_k = Σ_n γ_k(x_n) * (x_n-μ_k)(x_n-μ_k)^T / Σ_n γ_k(x_n)
            diff = pixels - new_means[k]  # [N, C]
            
            if self.covariance_type == 'diag':
                # 仅更新对角元素 (通道独立假设)
                weighted_var = torch.sum(resp_k.unsqueeze(1) * (diff ** 2), dim=0) / Nk
                new_covariances[k] = torch.diag(weighted_var.clamp(min=1e-4))
            else:
                # 完整协方差矩阵
                weighted_outer = torch.zeros(self.feature_dim, self.feature_dim, device=self.device)
                for i in range(min(N, 1000)):  # 限制计算量
                    if resp_k[i] > 1e-8:
                        outer = torch.outer(diff[i], diff[i])
                        weighted_outer += resp_k[i] * outer
                new_covariances[k] = weighted_outer / Nk + 1e-6 * torch.eye(self.feature_dim, device=self.device)
            
            # 更新权重: π_k = Σ_n γ_k(x_n) / N
            new_weights[k] = Nk / N
        
        # 更新参数
        self.means = new_means.detach()
        self.covariances = new_covariances.detach()
        self.weights = new_weights.detach()
    
    def fit(self, pixels, max_iters=None):
        """
        使用EM算法拟合GMM
        Args:
            pixels: 像素数据 [N, C]
            max_iters: 最大迭代次数 (覆盖初始化时的设置)
        Returns:
            assignments: 每个像素的分量分配 [N]
        """
        if not self.initialized:
            self._initialize_parameters(pixels)
        
        max_iters = max_iters or self.max_iters
        prev_log_likelihood = float('-inf')
        log_likelihoods = []
        
        print(f"[PixelGMM] Running EM algorithm for {max_iters} iterations...")
        for i in tqdm(range(max_iters), desc='EM Iterations'):
            # E步
            responsibilities, log_likelihood = self.e_step(pixels)
            log_likelihoods.append(log_likelihood)
            
            # 检查收敛
            if i > 0 and abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"EM converged at iteration {i+1}")
                break
            
            prev_log_likelihood = log_likelihood
            
            # M步
            self.m_step(pixels, responsibilities)
        
        # 最终分配: 每个像素分配到概率最高的分量
        assignments = torch.argmax(responsibilities, dim=1)  # [N]
        print(f"[PixelGMM] EM completed. Final log-likelihood: {log_likelihood:.4f}")
        print(f"Component assignments: {torch.bincount(assignments, minlength=self.num_components).cpu().numpy()}")
        
        return assignments
    
    def predict(self, pixels):
        """
        预测像素的分量分配
        Args:
            pixels: 像素数据 [N, C] 或 [B, C, H, W]
        Returns:
            assignments: 分量分配 [N] 或 [B, H, W]
            probabilities: 概率分布 [N, k] 或 [B, H, W, k]
        """
        # 处理图像格式输入
        if len(pixels.shape) == 4:  # [B, C, H, W]
            B, C, H, W = pixels.shape
            pixels_flat = pixels.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            is_image = True
        else:
            pixels_flat = pixels
            is_image = False
        
        # E步计算责任矩阵
        responsibilities, _ = self.e_step(pixels_flat)
        assignments_flat = torch.argmax(responsibilities, dim=1)
        
        # 恢复原始形状
        if is_image:
            assignments = assignments_flat.reshape(B, H, W)
            probabilities = responsibilities.reshape(B, H, W, self.num_components)
            return assignments, probabilities
        else:
            return assignments_flat, responsibilities
    
    def get_component_statistics(self, pixels, assignments):
        """
        计算每个GMM分量的风格统计量 (用于风格迁移)
        注意: 这些统计量不同于GMM参数!
        Args:
            pixels: 像素数据 [N, C]
            assignments: 像素分配 [N]
        Returns:
            component_stats: {
                'means': [k, C] - 每个分量的通道均值,
                'stds': [k, C] - 每个分量的通道标准差
            }
        """
        component_stats = {
            'means': torch.zeros(self.num_components, self.feature_dim, device=self.device),
            'stds': torch.zeros(self.num_components, self.feature_dim, device=self.device),
            'counts': torch.zeros(self.num_components, device=self.device)
        }
        
        for k in range(self.num_components):
            mask = (assignments == k)
            count_k = torch.sum(mask.float())
            component_stats['counts'][k] = count_k
            
            if count_k > 0:
                pixels_k = pixels[mask]
                # 通道级均值 (风格统计量)
                component_stats['means'][k] = torch.mean(pixels_k, dim=0)
                # 通道级标准差 (风格统计量，从数据计算，非从协方差矩阵提取)
                component_stats['stds'][k] = torch.std(pixels_k, dim=0) + 1e-8
        
        return component_stats
    
    def update_incremental(self, new_pixels, alpha=0.9):
        """
        增量更新GMM参数 (适用于在线学习)
        公式: new_param = alpha * old_param + (1-alpha) * new_param
        Args:
            new_pixels: 新像素数据 [N, C]
            alpha: 混合系数 (0.9表示90%保留历史)
        """
        if not self.initialized:
            self.fit(new_pixels, max_iters=20)
            return
        
        # 1. 为新像素分配分量
        assignments, _ = self.predict(new_pixels)
        
        # 2. 计算新统计量
        new_stats = self.get_component_statistics(new_pixels, assignments)

        # 1. 计算当前批次的总样本数 (所有簇的计数之和)
        # 注意: new_stats['counts'] 是一个 tensor 或 array，包含每个簇在这个 batch 里的像素数
        batch_total_count = torch.sum(new_stats['counts'])
        
        # 防止除以 0 (极少数情况)
        if batch_total_count < 1e-6:
            return  # 本次不做更新
        
        # 3. 指数移动平均更新
        for k in range(self.num_components):
            if new_stats['counts'][k] > 0:
                # 更新均值
                self.means[k] = alpha * self.means[k] + (1 - alpha) * new_stats['means'][k]
                
                # 更新协方差 (简化: 仅更新对角元素)
                if self.covariance_type == 'diag':
                    new_var = new_stats['stds'][k] ** 2
                    old_var = torch.diagonal(self.covariances[k])
                    updated_var = alpha * old_var + (1 - alpha) * new_var
                    self.covariances[k] = torch.diag(updated_var)
                
                # # 更新权重 (基于计数)
                # total_count = self.weights[k] * 1000 + new_stats['counts'][k]  # 假设历史有1000个样本
                # self.weights[k] = (self.weights[k] * 1000 * alpha + new_stats['counts'][k] * (1 - alpha)) / total_count
                # --- 步骤 A: 计算当前 Batch 中该簇的占比 ---
                # current_batch_weight 即 π_{k, batch}
                current_batch_weight = new_stats['counts'][k] / batch_total_count
                
                # --- 步骤 B: 动量更新 (EMA) ---
                # self.weights[k] 是历史积累的权重
                # alpha 是动量 (如 0.9), (1-alpha) 是学习率
                self.weights[k] = self.weights[k] * alpha + current_batch_weight * (1 - alpha)
        
        # 确保权重和为1
        # self.weights = self.weights / torch.sum(self.weights)
        # 3. 归一化 (Normalization) - 非常重要！
        # 经过多次独立更新后，权重之和可能略微偏离 1.0，必须重新归一化
        self.weights = self.weights / self.weights.sum()
        print(f"[PixelGMM] Incremental update completed with alpha={alpha}")