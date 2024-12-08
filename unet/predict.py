
import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None # 如果CUDA可用，执行CUDA同步以确保时间同步
    return time.time() # 返回当前时间


def main():
    classes = 1  # 类别数，不包括背景
    weights_path = "./save_weights/best_model.pth"
    # 图像路径和感兴趣区域掩码路径
    img_path = "./DRIVE/test/images/01_test.tif"
    roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"
    # 确保权重文件、图像文件和感兴趣区域掩码文件存在
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."
    # 图像的均值和标准差
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 获取设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 创建模型
    model = UNet(in_channels=3, num_classes=classes+1, base_c=32)

    # 加载权重
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # 加载感兴趣区域掩码
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)

    # 加载图像
    original_img = Image.open(img_path).convert('RGB')

    # 从PIL图像转换为张量并进行标准化
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # 扩展批次维度
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # 初始化模型，运行一次以便加载模型
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)
        # 开始推理
        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))
        # 获取预测结果并后处理
        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不感兴趣的区域像素设置成0(黑色)
        prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("test_result.png")


if __name__ == '__main__':
    main()
