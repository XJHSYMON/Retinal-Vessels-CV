# U-Net(Convolutional Networks for Biomedical Image Segmentation)

## 环境配置：
* Pytorch>=1.10
* Ubuntu或Centos或Windows(Windows暂不支持多GPU训练)
* 最好使用GPU训练
* 详细环境配置见`requirements.txt`

## 文件结构：
```
  ├── src: 搭建U-Net模型代码
  ├── train_utils: 训练、验证以及多GPU训练相关模块
  ├── my_dataset.py: 自定义dataset用于读取DRIVE数据集(视网膜血管分割)
  ├── train.py: 以单GPU为例进行训练
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  └── compute_mean_std.py: 统计数据集各通道的均值和标准差
```


## 训练方法
* 确保提前准备好数据集
* 若要使用单GPU或者CPU训练，直接使用train.py训练脚本
* 若要使用多GPU训练，使用`torchrun --nproc_per_node=8 train_multi_GPU.py`指令,`nproc_per_node`参数为使用GPU数量
* 如果想指定使用哪些GPU设备可在指令前加上`CUDA_VISIBLE_DEVICES=0,3`(例如我只要使用设备中的第1块和第4块GPU设备)
* `CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py`

## 推理方法
* 请先训练完成获得模型的权重文件
* 运行predict.py，确保图片路径正确

## 注意事项
* 在使用训练脚本时，注意要将`--data-path`设置为自己存放`DRIVE`文件夹所在的**根目录**
* 在使用预测脚本时，要将`weights_path`设置为你自己生成的权重路径。
* 使用validation文件时，注意确保你的验证集或者测试集中必须包含每个类别的目标，并且使用时只需要修改`--num-classes`、`--data-path`和`--weights`即可，其他代码尽量不要改动
