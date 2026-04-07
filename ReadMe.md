1. 基本训练命令
```python
python main.py --source_domain amazon --target_domain dslr --num_gaussians 5 --batch_size 32 --epochs 50 --device cuda:1 --checkpoint_dir ./checkpoints --log_dir ./logs
```
2. 典型实验配置
```python
# 实验1: Amazon → DSLR (标准配置)
python main.py --source_domain amazon，webcam --target_domain dslr --num_gaussians 5 --device cuda:1

# 实验2: Amazon → Webcam (更具挑战性)
python main.py --source_domain amazon --target_domain webcam --num_gaussians 7 --device cuda:1

# 实验3: 小批量快速验证 (调试用)
python main.py --source_domain amazon --target_domain dslr --batch_size 16 --epochs 10 --device cuda:1
```
3. 模型评估
训练完成后，使用以下命令评估最佳模型：
```python
python evaluate.py --checkpoint ./checkpoints/best_model.pth --source_domain amazon --target_domain dslr --device cuda:1
```

1. 基本训练命令
```bash
python main.py \
	--source_domain amazon \
	--target_domain dslr \
	--num_gaussians 5 \
	--batch_size 32 \
	--epochs 50 \
	--device cuda:1 \
	--checkpoint_dir ./checkpoints \
	--log_dir ./logs
```

2. 三种风格模式（新增开关）
```bash
# 仅像素级风格迁移
python main.py --source_domain amazon --target_domain dslr --style_mode pixel --lambda_pixel 1.0

Loading datasets...
Source train: 2253 images
Source val: 564 images
Target train (unlabeled): 498 images
Target test: 498 images

Creating model...
/home/amax/anaconda3/envs/gmm-da/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/amax/anaconda3/envs/gmm-da/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Creating domain adapter...

Starting training...

==================================================
Epoch 1/10
==================================================
Epoch 1:   0%|                                                                                                                     | 0/71 [00:00<?, ?it/s]
[Train] Lazy initializing GMM with the first target batch...
[PixelGMM] Initializing parameters using K-Means (k=5)...
[PixelGMM] Initialization complete on cuda:0
[PixelGMM] Running EM algorithm for 20 iterations...
EM Iterations: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [11:22<00:00, 34.14s/it]
[PixelGMM] EM completed. Final log-likelihood: 3689561.7500███████████████████████████████████████████████████████████████| 20/20 [11:22<00:00, 33.39s/it]
Component assignments: [137426 362507 328664 506105 270930]
[Train] GMM initialized successfully.
Epoch 1: 100%|███████████████████████████████████████████████████████| 71/71 [53:16<00:00, 45.03s/it, Loss=4.3041, Acc=48.91%, a=0.80, Lp=1.345, Lf=0.000]
Validating source: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:01<00:00, 16.99it/s]
Source domain accuracy: 79.96%

==================================================
Epoch 2/10
==================================================
Epoch 2: 100%|███████████████████████████████████████████████████████| 71/71 [54:17<00:00, 45.89s/it, Loss=1.4881, Acc=81.94%, a=0.80, Lp=0.970, Lf=0.000]
Validating source: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 26.04it/s]
Source domain accuracy: 84.75%

==================================================
Epoch 3/10
==================================================
Epoch 3: 100%|███████████████████████████████████████████████████████| 71/71 [44:26<00:00, 37.56s/it, Loss=0.9160, Acc=88.50%, a=0.80, Lp=0.495, Lf=0.000]
Validating source: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 24.86it/s]
Source domain accuracy: 86.52%

==================================================
Epoch 4/10
==================================================
Epoch 4: 100%|███████████████████████████████████████████████████████| 71/71 [47:47<00:00, 40.38s/it, Loss=0.5870, Acc=93.52%, a=0.80, Lp=0.169, Lf=0.000]
Validating source: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 25.74it/s]
Source domain accuracy: 87.94%

==================================================
Epoch 5/10
==================================================
Epoch 5: 100%|███████████████████████████████████████████████████████| 71/71 [40:14<00:00, 34.01s/it, Loss=0.4108, Acc=96.01%, a=0.80, Lp=0.861, Lf=0.000]
Validating source: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 26.67it/s]
Source domain accuracy: 87.59%
Validating target: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:01<00:00,  8.53it/s]
Target domain accuracy: 80.72%
✓ New best model saved: 80.72%

==================================================
Epoch 6/10
==================================================
Epoch 6: 100%|███████████████████████████████████████████████████████| 71/71 [40:00<00:00, 33.81s/it, Loss=0.2698, Acc=97.69%, a=0.80, Lp=0.466, Lf=0.000]
Validating source: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 25.21it/s]
Source domain accuracy: 88.48%

==================================================
Epoch 7/10
==================================================
Epoch 7: 100%|███████████████████████████████████████████████████████| 71/71 [41:41<00:00, 35.23s/it, Loss=0.1745, Acc=98.62%, a=0.80, Lp=0.070, Lf=0.000]
Validating source: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 25.56it/s]
Source domain accuracy: 87.94%

==================================================
Epoch 8/10
==================================================
Epoch 8: 100%|███████████████████████████████████████████████████████| 71/71 [40:00<00:00, 33.81s/it, Loss=0.1289, Acc=99.20%, a=0.80, Lp=0.033, Lf=0.000]
Validating source: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 26.45it/s]
Source domain accuracy: 87.06%

==================================================
Epoch 9/10
==================================================
Epoch 9: 100%|███████████████████████████████████████████████████████| 71/71 [41:30<00:00, 35.08s/it, Loss=0.1041, Acc=99.42%, a=0.80, Lp=0.393, Lf=0.000]
Validating source: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 26.36it/s]
Source domain accuracy: 89.01%

==================================================
Epoch 10/10
==================================================
Epoch 10: 100%|██████████████████████████████████████████████████████| 71/71 [39:30<00:00, 33.38s/it, Loss=0.0920, Acc=99.78%, a=0.80, Lp=0.482, Lf=0.000]
Validating source: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 24.77it/s]
Source domain accuracy: 88.65%
Validating target: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:01<00:00,  9.00it/s]
Target domain accuracy: 80.32%

==================================================
Training completed. Best target accuracy: 80.72%
==================================================

# 仅特征级风格迁移（WCT + 软分配）
python main.py --source_domain amazon --target_domain dslr --style_mode feature --lambda_feature 1.0

# 双分支（像素级 + 特征级）
python main.py --source_domain amazon --target_domain dslr --style_mode both --lambda_pixel 1.0 --lambda_feature 1.0
```

3. 自动选 K（新增开关）
```bash
# 开启 BIC 自动选 K，候选集合可自定义
python main.py \
	--source_domain amazon \
	--target_domain dslr \
	--auto_select_k \
	--bic_k_candidates 3,5,7,9 \
	--bic_num_batches 1 \
	--bic_em_iters 15
```

4. 风格强度起止参数（可调）
```bash
# 在一个 epoch 内按 batch 进度从 start 线性过渡到 end
python main.py \
	--source_domain amazon \
	--target_domain dslr \
	--style_alpha_start 0.2 \
	--style_alpha_end 0.9
```

5. 组合示例（推荐）
```bash
python main.py \
	--source_domain art_painting,cartoon,photo \
	--target_domain  sketch\
	--style_mode both \
	--lambda_pixel 1.0 \
	--lambda_feature 0.5 \
	--auto_select_k \
	--bic_k_candidates 3,5,7 \
	--bic_num_batches 1 \
	--bic_em_iters 15 \
	--style_alpha_start 0.3 \
	--style_alpha_end 0.8
```

6. 模型评估
训练完成后，使用以下命令评估最佳模型：
```bash
python evaluate.py --checkpoint ./checkpoints/best_model.pth --source_domain amazon --target_domain dslr --device cuda:1
```

