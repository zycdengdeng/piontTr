#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
车辆点云补全推理脚本
根据 CSV 文件筛选 Car, SUV, Truck, Bus 类别进行推理
"""

import argparse
import os
import numpy as np
import pandas as pd
import cv2
import sys
from tqdm import tqdm
import open3d as o3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from tools import builder
from utils.config import cfg_from_yaml_file
from utils import misc
from datasets.io import IO
from datasets.data_transforms import Compose


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_config',
        type=str,
        default='cfgs/KITTI_models/PoinTr.yaml',
        help='yaml config file')
    parser.add_argument(
        '--model_checkpoint',
        type=str,
        default='data_preparation/PoinTr_KITTI.pth',
        help='pretrained weight')
    parser.add_argument(
        '--pc_root',
        type=str,
        default='/mnt/zyc_wzh/dyn_merged_final_1',
        help='输入点云文件夹路径')
    parser.add_argument(
        '--csv_path',
        type=str,
        default='data_preparation/unique_id_statistics.csv',
        help='ID-Label 映射 CSV 文件路径')
    parser.add_argument(
        '--out_pc_root',
        type=str,
        default='output/completed_vehicles',
        help='输出点云文件夹路径')
    parser.add_argument(
        '--save_vis_img',
        action='store_true',
        default=False,
        help='是否保存可视化图片')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='推理设备')
    parser.add_argument(
        '--target_labels',
        nargs='+',
        default=['Car', 'Suv', 'Truck', 'Bus'],
        help='需要推理的车辆类别')
    args = parser.parse_args()
    return args


def load_vehicle_ids(csv_path, target_labels):
    """
    从 CSV 文件读取并筛选出目标类别的 ID

    Args:
        csv_path: CSV 文件路径
        target_labels: 目标类别列表，如 ['Car', 'Suv', 'Truck', 'Bus']

    Returns:
        valid_ids: 符合条件的 ID 集合
    """
    df = pd.read_csv(csv_path)
    # 筛选出目标类别
    valid_df = df[df['Label'].isin(target_labels)]
    valid_ids = set(valid_df['ID'].values)

    print(f"从 {csv_path} 读取到 {len(df)} 个 ID")
    print(f"目标类别: {target_labels}")
    print(f"符合条件的 ID 数量: {len(valid_ids)}")
    print(f"符合条件的 ID: {sorted(valid_ids)}")

    return valid_ids


def get_id_from_filename(filename):
    """
    从文件名提取 ID
    文件名格式: {id}_{timestamp}_{count}.pcd
    例如: 9_1742877043043_1.pcd -> 9

    Args:
        filename: 文件名

    Returns:
        id: 物体 ID，如果无法解析则返回 None
    """
    try:
        id_str = filename.split('_')[0]
        return int(id_str)
    except:
        return None


def filter_pc_files(pc_root, valid_ids):
    """
    筛选出符合条件的点云文件

    Args:
        pc_root: 点云文件夹路径
        valid_ids: 有效的 ID 集合

    Returns:
        filtered_files: 符合条件的文件名列表
    """
    all_files = os.listdir(pc_root)
    filtered_files = []

    for filename in all_files:
        if not filename.endswith('.pcd'):
            continue

        obj_id = get_id_from_filename(filename)
        if obj_id is not None and obj_id in valid_ids:
            filtered_files.append(filename)

    print(f"\n总点云文件数: {len(all_files)}")
    print(f"符合条件的文件数: {len(filtered_files)}")

    return filtered_files


def inference_single(model, pc_path, args, config, root=None):
    """
    对单个点云文件进行推理
    """
    if root is not None:
        pc_file = os.path.join(root, pc_path)
    else:
        pc_file = pc_path

    # 读取点云
    pc_ndarray = IO.get(pc_file).astype(np.float32)

    # 根据模型类型进行归一化
    if config.dataset.train._base_['NAME'] == 'ShapeNet':
        centroid = np.mean(pc_ndarray, axis=0)
        pc_ndarray = pc_ndarray - centroid
        m = np.max(np.sqrt(np.sum(pc_ndarray**2, axis=1)))
        pc_ndarray = pc_ndarray / m

    # 数据变换
    transform = Compose([{
        'callback': 'UpSamplePoints',
        'parameters': {
            'n_points': 2048
        },
        'objects': ['input']
    }, {
        'callback': 'ToTensor',
        'objects': ['input']
    }])

    pc_ndarray_normalized = transform({'input': pc_ndarray})

    # 推理
    ret = model(pc_ndarray_normalized['input'].unsqueeze(0).to(args.device.lower()))
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()

    # 反归一化
    if config.dataset.train._base_['NAME'] == 'ShapeNet':
        dense_points = dense_points * m
        dense_points = dense_points + centroid

    # 保存结果
    if args.out_pc_root != '':
        # 使用原文件名（不含扩展名）作为子文件夹名
        target_path = os.path.join(args.out_pc_root, os.path.splitext(pc_path)[0])
        os.makedirs(target_path, exist_ok=True)

        # 保存补全后的点云为 PCD 格式
        pcd_completed = o3d.geometry.PointCloud()
        pcd_completed.points = o3d.utility.Vector3dVector(dense_points)
        o3d.io.write_point_cloud(os.path.join(target_path, 'completed.pcd'), pcd_completed)

        # 保存原始点云为 PCD 格式（用于对比）
        pcd_input = o3d.geometry.PointCloud()
        pcd_input.points = o3d.utility.Vector3dVector(pc_ndarray)
        o3d.io.write_point_cloud(os.path.join(target_path, 'input.pcd'), pcd_input)

        # 保存可视化图片
        if args.save_vis_img:
            input_img = misc.get_ptcloud_img(pc_ndarray_normalized['input'].numpy())
            dense_img = misc.get_ptcloud_img(dense_points)
            cv2.imwrite(os.path.join(target_path, 'input.jpg'), input_img)
            cv2.imwrite(os.path.join(target_path, 'completed.jpg'), dense_img)

    return dense_points


def main():
    args = get_args()

    print("="*60)
    print("车辆点云补全推理")
    print("="*60)
    print(f"配置文件: {args.model_config}")
    print(f"模型权重: {args.model_checkpoint}")
    print(f"输入目录: {args.pc_root}")
    print(f"输出目录: {args.out_pc_root}")
    print(f"CSV 文件: {args.csv_path}")
    print(f"目标类别: {args.target_labels}")
    print("="*60)

    # 加载配置
    config = cfg_from_yaml_file(args.model_config)

    # 构建模型
    print("\n正在加载模型...")
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.model_checkpoint)
    base_model.to(args.device.lower())
    base_model.eval()
    print("模型加载完成!")

    # 读取 CSV 并筛选 ID
    print("\n正在读取 CSV 文件...")
    valid_ids = load_vehicle_ids(args.csv_path, args.target_labels)

    # 筛选点云文件
    print("\n正在筛选点云文件...")
    filtered_files = filter_pc_files(args.pc_root, valid_ids)

    if len(filtered_files) == 0:
        print("\n警告: 没有找到符合条件的点云文件!")
        return

    # 创建输出目录
    os.makedirs(args.out_pc_root, exist_ok=True)

    # 批量推理
    print(f"\n开始推理 {len(filtered_files)} 个点云文件...")
    print("="*60)

    for pc_file in tqdm(filtered_files, desc="推理进度"):
        try:
            inference_single(base_model, pc_file, args, config, root=args.pc_root)
        except Exception as e:
            print(f"\n处理 {pc_file} 时出错: {e}")
            continue

    print("\n" + "="*60)
    print("推理完成!")
    print(f"结果保存在: {args.out_pc_root}")
    print("="*60)


if __name__ == '__main__':
    main()
