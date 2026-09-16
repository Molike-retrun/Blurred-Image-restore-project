# 基于 CNN、残差 CNN 与 U-Net 的图像恢复项目

## 运行方式
首先安装项目依赖：
pip install -r requirements.txt
然后运行：
python main.py
程序会依次完成：
1. 加载配置
2. 设置随机种子
3. 构建 Dataset 与 DataLoader
4. 训练三个模型
5. Early Stopping
6. 保存最佳 checkpoint
7. 自动模型排名
8. 选择最佳模型
9. 绘制训练曲线
10. 进行案例分析
11. 保存日志与可视化结果

## 1. 项目概述

本项目实现了一个基于深度学习的图像恢复流程，用于处理合成高斯模糊图像。

项目首先从清晰图像出发，通过高斯模糊生成退化图像，然后分别训练三种神经网络模型对图像进行恢复：

- Simple CNN
- Residual CNN
- 轻量级 U-Net

本项目重点关注以下内容：

- 图像恢复基础流程搭建
- 多模型对比
- 固定随机种子与可复现训练
- Early Stopping
- Best Checkpoint 保存
- 自动模型选择
- PSNR / SSIM / MAE / RMSE 等指标评价
- Best / Normal / Bad Case 案例分析
- Logging 与训练过程记录

---

## 2. 任务定义

本项目的图像恢复流程为：

清晰图像  
→ 高斯模糊退化  
→ 图像恢复网络  
→ 恢复图像

训练目标是使恢复图像尽可能接近原始清晰图像。

---

## 3. 数据集处理

当前阶段使用训练集和验证集。

每张图像经过以下处理：

1. 使用 OpenCV 读取图像
2. 将 BGR 格式转换为 RGB
3. 将图像缩放至 256 × 256
4. 将像素值归一化至 [0, 1]
5. 使用高斯模糊动态生成退化图像

当前阶段中的模糊图像不是提前存储的，而是在 Dataset 中根据清晰图像动态生成。

---

## 4. 图像退化模型

当前阶段使用高斯模糊作为图像退化模型。

当前参数为：

- Kernel Size：7 × 7
- Sigma：2

本阶段主要目标是完成图像恢复完整流程与模型对比，因此暂时使用固定的合成高斯模糊。

基于 PSF 的物理退化模型与 Zemax 光学仿真将在后续阶段继续扩展。

---

## 5. 模型结构

### 5.1 Simple CNN

Simple CNN 使用多层卷积直接学习：

模糊图像 → 清晰图像

作为本项目中的基础恢复模型。

### 5.2 Residual CNN

Residual CNN 使用残差学习思想：

Output = Input + F(Input)

其中 F(Input) 学习模糊图像与清晰图像之间的修正信息。

相比直接预测整张清晰图像，残差学习可以让网络更专注于学习模糊所造成的差异。

### 5.3 U-Net

本项目实现了一个轻量级 U-Net。

其主要结构包括：

- Encoder
- Bottleneck
- Decoder
- Skip Connection

通过跳跃连接，将编码阶段的特征与解码阶段特征进行融合，从而保留更多局部和空间信息。

---

## 6. 训练设置

当前训练设置包括：

- 损失函数：MSE Loss
- 优化器：Adam
- Early Stopping：启用
- 最优模型保存指标：Validation PSNR
- 随机种子：固定
- cuDNN deterministic：启用

固定随机种子用于提升实验的可复现性。

Early Stopping 用于防止模型在验证集上持续训练但性能不再提升。

---

## 7. 评价指标

项目当前使用以下指标评价模型：

- Validation Loss
- MAE
- RMSE
- PSNR
- SSIM

其中：

PSNR 被作为当前阶段最主要的模型选择指标。

模型训练过程中，当 Validation PSNR 达到新的最佳值时，会保存对应的模型 checkpoint。

---

## 8. 自动模型选择

每个模型分别保存自己的最佳 checkpoint：

- Simple CNN
- Residual CNN
- U-Net

然后根据各模型的最佳 Validation PSNR 自动进行排序。

最终选择 Validation PSNR 最高的模型作为当前阶段的最佳模型。

在当前实验设置下，Residual CNN 的综合表现最好。

---

## 9. 案例分析

在选出最佳模型后，对验证集中的样本进一步进行逐图分析。

根据：

Restored PSNR - Blurred PSNR

即恢复后的 PSNR 相对于原始模糊图像 PSNR 的提升程度，将样本分为：

- Best Case
- Normal Case
- Bad Case

其中：

- Best Case：PSNR 提升最大
- Normal Case：PSNR 提升处于中间位置
- Bad Case：PSNR 提升最小

此外还对以下图像特征进行分析：

- Laplacian Edge Strength
- Mean Brightness
- Blurred Edge Error
- Restored Edge Error
- Blurred Brightness Error
- Restored Brightness Error

用于观察模型除了提升 PSNR 之外，对图像边缘信息与整体亮度的恢复情况。

---

## 10. 项目结构

```text
CV_Image_Restoration/
├── config/
│   ├── config.yaml
│   └── config_loader.py
│
├── data/
│   ├── train/
│   │   └── clean/
│   └── val/
│       └── clean/
│
├── checkpoints/
│
├── logs/
│
├── results/
│
├── docs/
│   └── stage1_experiment.md
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train_one_epoch.py
│   ├── train_model.py
│   ├── validation.py
│   ├── calculate.py
│   ├── training_curve.py
│   ├── model_selection.py
│   ├── case_analysis.py
│   ├── image_analysis.py
│   ├── visualization.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md