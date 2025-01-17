# 方嘉彬+2022214636+智能应用系统设计
题目序号 3  
任务名称：图像的卡通、动漫化风格生成系统

这里是CycleGANpro的模型训练代码

## 环境搭建
### 1. 创建conda环境，实验在python==3.11下进行
```
conda create -n CycleGANpro python==3.11
conda activate CycleGANpro
```
### 2. 实验在torch2.1.0+CUDA12.2下进行
```
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```
### 3. 按照requirements.txt安装
```
pip install -r requirements.txt
```

## 数据集
实验使用selfie2anime数据集  
[selfie2anime](https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view?usp=drive_link)  
```
path_to_Repo/
    selfie2anime/
        ├── testA/
        ├── testB/
        ├── trainA/
        └── trainB/
```

## 预训练模型
预训练模型可在以下链接下载  
[模型权重](https://drive.google.com/file/d/1svWqkziH3sECWHlRGUS3R_SOrCkBZiAf/view?usp=drive_link)

## 模型训练
```
python train.py
```

## 系统展示
系统展示代码请移步[StyleTransfer](https://github.com/para133/StyleTransfer)

## 参考的仓库  
本项目参考了以下仓库：
- [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
