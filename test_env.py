#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境配置验证脚本
用于检查 PoinTr 项目所需的所有依赖是否正确安装
"""

import sys

def test_pytorch():
    """测试 PyTorch 和 CUDA"""
    print("\n" + "="*50)
    print("测试 PyTorch 和 CUDA")
    print("="*50)
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("✗ 警告: CUDA 不可用，无法使用 GPU")
            return False
        return True
    except ImportError as e:
        print(f"✗ PyTorch 未安装: {e}")
        return False

def test_basic_dependencies():
    """测试基础 Python 依赖"""
    print("\n" + "="*50)
    print("测试基础依赖包")
    print("="*50)

    dependencies = [
        'numpy',
        'scipy',
        'h5py',
        'yaml',
        'easydict',
        'tqdm',
        'open3d',
        'cv2',
        'timm',
        'tensorboardX',
        'matplotlib',
        'einops',
        'transforms3d'
    ]

    all_ok = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError as e:
            print(f"✗ {dep} - {e}")
            all_ok = False

    return all_ok

def test_cuda_extensions():
    """测试 CUDA 扩展模块"""
    print("\n" + "="*50)
    print("测试 CUDA 扩展模块")
    print("="*50)

    all_ok = True

    # Chamfer Distance
    try:
        import chamfer_dist
        print("✓ Chamfer Distance")
    except ImportError as e:
        print(f"✗ Chamfer Distance - {e}")
        all_ok = False

    # Gridding
    try:
        import gridding
        print("✓ Gridding")
    except ImportError as e:
        print(f"✗ Gridding - {e}")
        all_ok = False

    # Gridding Loss
    try:
        import gridding_loss
        print("✓ Gridding Loss")
    except ImportError as e:
        print(f"✗ Gridding Loss - {e}")
        all_ok = False

    # Cubic Feature Sampling
    try:
        import cubic_feature_sampling
        print("✓ Cubic Feature Sampling")
    except ImportError as e:
        print(f"✗ Cubic Feature Sampling - {e}")
        all_ok = False

    # PointNet++
    try:
        import pointnet2_ops
        print("✓ PointNet++")
    except ImportError as e:
        print(f"✗ PointNet++ - {e}")
        print("  安装命令: pip install \"git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib\"")
        all_ok = False

    return all_ok

def test_simple_forward():
    """测试简单的前向传播"""
    print("\n" + "="*50)
    print("测试简单的 GPU 计算")
    print("="*50)

    try:
        import torch
        if not torch.cuda.is_available():
            print("⊘ 跳过（无 GPU）")
            return True

        # 创建一个简单的张量并移到 GPU
        x = torch.randn(2, 3, 224, 224).cuda()
        y = x * 2 + 1
        print(f"✓ GPU 计算测试通过")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {y.shape}")
        print(f"  使用设备: {x.device}")
        return True
    except Exception as e:
        print(f"✗ GPU 计算失败: {e}")
        return False

def main():
    """主函数"""
    print("\n" + "="*50)
    print("PoinTr 环境配置检查")
    print("="*50)

    results = {
        'PyTorch & CUDA': test_pytorch(),
        '基础依赖': test_basic_dependencies(),
        'CUDA 扩展': test_cuda_extensions(),
        'GPU 计算': test_simple_forward()
    }

    print("\n" + "="*50)
    print("测试总结")
    print("="*50)

    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*50)
    if all_passed:
        print("✓ 所有测试通过！环境配置成功！")
        print("="*50)
        print("\n下一步：")
        print("1. 下载数据集（参考 DATASET.md）")
        print("2. 下载预训练模型（参考 README.md）")
        print("3. 运行推理或训练")
        return 0
    else:
        print("✗ 部分测试失败，请检查上述错误信息")
        print("="*50)
        print("\n建议：")
        print("1. 查看 SETUP_GUIDE.md 获取详细配置步骤")
        print("2. 检查 CUDA 版本是否正确")
        print("3. 重新运行 install.sh 安装扩展模块")
        return 1

if __name__ == '__main__':
    sys.exit(main())
