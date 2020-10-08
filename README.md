# SRP项目--基于轻量化网络的图像分类

## 项目结构

```console
.
├─data：数据集
│   ├─test1 //测试集
│   ├─train //（被划分为训练集和验证集）
│   │   ├─domain1
│   │   │   ├─XML // 标注文件
│   │   ├─domain2
│   │   │   ├─XML // 标注文件
│   │   ├─domain3
│   │   │   ├─XML // 标注文件
│   │   ├─domain4
│   │   │   ├─XML // 标注文件
│   │   ├─domain5
│   │   │   ├─XML // 标注文件
│   │   ├─domain6
│   │   │   ├─XML // 标注文件
├─README.md
├─datasets.py // 提取、处理数据集
├─draw.py // 绘制数据集
├─model.py // 一个非常简单的CNN网络
├─random_crop.py // 将原始图片随机分割，以获取normal分类
├─training.py // 主程序

```

## 运行

**开始训练**

```bash
python ./training.py
```

### 参数列表

- --num-workers 工作进程数
- --batch-size 每批次大小
- --epochs 迭代次数
