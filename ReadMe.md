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
	--source_domain amazon \
	--target_domain dslr \
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

