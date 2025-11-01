# PoinTr 项目环境配置指南

本指南适用于 CUDA 11.8 环境。

## 环境要求

- Python >= 3.7
- CUDA 11.8
- GCC >= 4.9
- PyTorch >= 1.7.0

## 配置步骤

### 1. 安装 PyTorch (CUDA 11.8)

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

**验证安装：**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. 安装基础 Python 依赖

```bash
pip install -r requirements.txt
```

**注意事项：**
- 如果 `open3d==0.9` 安装失败，可以尝试安装最新版本：`pip install open3d`
- 如果 `timm==0.4.5` 安装失败，可以尝试：`pip install timm`

### 3. 编译 CUDA 扩展模块

#### 3.1 Chamfer Distance 和其他扩展

```bash
bash install.sh
```

这会安装以下扩展：
- Chamfer Distance
- Cubic Feature Sampling
- Gridding & Gridding Reverse
- Gridding Loss

**常见问题：** 如果遇到编译错误，参考 [Issue #6](https://github.com/yuxumin/PoinTr/issues/6)

#### 3.2 PointNet++

```bash
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

#### 3.3 GPU kNN

```bash
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

### 4. 手动安装扩展（如果步骤3失败）

如果遇到 `ModuleNotFoundError: No module named 'gridding'` 等错误：

```bash
cd extensions/chamfer_dist
python setup.py install

cd ../cubic_feature_sampling
python setup.py install

cd ../gridding
python setup.py install

cd ../gridding_loss
python setup.py install

cd ../..
```

### 5. 验证环境配置

创建测试脚本 `test_env.py`：

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# 测试扩展模块
try:
    import chamfer_dist
    print("✓ Chamfer Distance imported")
except ImportError as e:
    print(f"✗ Chamfer Distance failed: {e}")

try:
    import gridding
    print("✓ Gridding imported")
except ImportError as e:
    print(f"✗ Gridding failed: {e}")

try:
    import pointnet2_ops
    print("✓ PointNet++ imported")
except ImportError as e:
    print(f"✗ PointNet++ failed: {e}")
```

运行测试：
```bash
python test_env.py
```

## 可能遇到的问题

### 问题 1: CUDA 版本不匹配

**症状：** PyTorch 提示 CUDA 版本不匹配
**解决：** 确保 CUDA 11.8 的 nvcc 在 PATH 中：

```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

可以添加到 `~/.bashrc` 或 `~/.zshrc`

### 问题 2: GCC 版本问题

**症状：** 编译扩展时报 GCC 版本错误
**解决：** 确保 GCC >= 4.9，查看版本：

```bash
gcc --version
```

### 问题 3: 扩展模块找不到

**症状：** `ModuleNotFoundError: No module named 'gridding'`
**解决：** 进入对应的 extensions 目录手动安装（见步骤4）

### 问题 4: open3d 安装失败

**症状：** open3d==0.9 在新版 Python 上无法安装
**解决：**
```bash
pip install open3d  # 安装最新版本
```

## 下一步

环境配置完成后，您可以：

1. **下载数据集**：参考 [DATASET.md](./DATASET.md)
2. **下载预训练模型**：从 README 中的链接下载
3. **运行推理**：
   ```bash
   python tools/inference.py cfgs/PCN_models/PoinTr.yaml ckpts/PoinTr_PCN.pth --pc_root demo/ --out_pc_root inference_result/
   ```
4. **训练模型**：
   ```bash
   bash ./scripts/train.sh 0 --config ./cfgs/PCN_models/PoinTr.yaml --exp_name my_experiment
   ```

## 快速测试命令（完整流程）

```bash
# 1. 安装 PyTorch
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 2. 安装依赖
pip install -r requirements.txt

# 3. 编译扩展
bash install.sh

# 4. 安装 PointNet++ 和 kNN
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# 5. 验证
python test_env.py
```

---

**如有问题，请查看：**
- [官方 README](./README.md)
- [GitHub Issues](https://github.com/yuxumin/PoinTr/issues)
