1. 基本训练命令
```python
python main.py --source_domain amazon --target_domain dslr --num_gaussians 5 --batch_size 32 --epochs 50 --device cuda:1 --checkpoint_dir ./checkpoints --log_dir ./logs
```
2. 典型实验配置
```python
# 实验1: Amazon → DSLR (标准配置)
python main.py --source_domain amazon --target_domain dslr --num_gaussians 5 --device cuda:1

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
