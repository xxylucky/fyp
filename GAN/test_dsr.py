"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils
from PIL import Image, ImageDraw
import numpy as np
import os
import models.dcgan as dcgan
# import sys
# sys.path.append('/root/example/airs1/GAN') # 把代码根目录加入路径
# import models.dcgan as dcgan

def test_discriminator(model_path, output_dir='./test_results'):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
def test_discriminator(model_path):
    # 参数必须与你训练时 main.py 中的 netD 初始化参数完全一致
    # opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers
    imageSize = 64
    nz = 100
    nc = 1       # Mask 是单通道
    ndf = 64     # 你的隐藏层通道数
    ngpu = 1
    n_extra_layers = 0 # 检查你 opt 里的默认值，通常是 0

    # 实例化模型
    model = dcgan.DCGAN_D(imageSize, nz, nc, ndf, ngpu, n_extra_layers)
    
    # 加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)  
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    

    print(f"成功加载模型: {model_path}")
    print("-" * 50)

    # 2. 构造三种测试样本 (64x64)
    # A: 模拟一个健康的圆形 (假装是正常的解剖结构)
    real_shape = Image.new('L', (64, 64), 0)
    draw = ImageDraw.Draw(real_shape)
    draw.ellipse([15, 15, 45, 45], fill=255)
    
    # B: 模拟一个严重畸形的形状 (比如中间挖个洞，边缘破碎)
    bad_shape = Image.new('L', (64, 64), 0)
    draw_bad = ImageDraw.Draw(bad_shape)
    draw_bad.rectangle([10, 10, 50, 50], fill=255) # 先画个方块
    draw_bad.ellipse([25, 25, 35, 35], fill=0)    # 中间挖个大洞（解剖学不合理）

    # 3. 预处理函数
    def prepare_tensor(img):
        t = torch.from_numpy(np.array(img)).float() / 255.0
        t = (t - 0.5) / 0.5  # 对应训练时的归一化 [-1, 1]
        return t.view(1, 1, 64, 64).to(device)

    t_real = prepare_tensor(real_shape)
    t_bad = prepare_tensor(bad_shape)

    # 4. 判别器打分
    with torch.no_grad():
        score_real = model(t_real).item()
        score_bad = model(t_bad).item()

    # 5. 输出结果
    print(f"【正常形状】得分: {score_real:.4f}")
    print(f"【畸形形状】得分: {score_bad:.4f}")
    print("-" * 50)

    if score_real > score_bad:
        print("✅ 测试通过：判别器能够识别解剖学上的不合理形状。")
        print(f"分差 (Margin): {score_real - score_bad:.4f}")
    else:
        print("❌ 测试失败：判别器无法区分形状好坏，可能训练不足或模式坍塌。")

    # 保存对比图查看
    vutils.save_image(torch.cat([t_real.cpu(), t_bad.cpu()], 0), 
                      f"{output_dir}/comparison.png", normalize=True)
    print(f"对比图已保存至: {output_dir}/comparison.png")

if __name__ == "__main__":
    # 修改为你最新的权重路径
    PATH = "/root/example/airs1/semi/code/pretrain/GAN/netD_epoch_10000.pth" 
    if os.path.exists(PATH):
        test_discriminator(PATH)
    else:
        print(f"错误：找不到模型文件 {PATH}")"""

import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image, ImageDraw
import numpy as np
import os
import models.dcgan as dcgan

def test_discriminator(model_path, output_dir='./test_results'):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # 参数与 main.py 保持一致
    imageSize = 64
    nz = 100
    nc = 1
    ndf = 64
    ngpu = 1
    n_extra_layers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dcgan.DCGAN_D(imageSize, nz, nc, ndf, ngpu, n_extra_layers)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"\n>>> 成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 构造测试样本
    # 1. 正常的椭圆
    real_shape = Image.new('L', (64, 64), 0)
    draw = ImageDraw.Draw(real_shape)
    draw.ellipse([20, 20, 45, 45], fill=255)
    
    # 2. 极其离谱的畸形 (散点 + 细长条)
    bad_shape = Image.new('L', (64, 64), 0)
    draw_bad = ImageDraw.Draw(bad_shape)
    draw_bad.point([(5,5), (55,5), (5,55), (55,55), (32,32)], fill=255)
    draw_bad.line([0, 0, 64, 64], fill=255, width=1)

    def to_tensor(img):
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr).view(1, 1, 64, 64).to(device)

    t_real = to_tensor(real_shape)
    t_bad = to_tensor(bad_shape)

    with torch.no_grad():
        score_real = model(t_real).mean().item()
        score_bad = model(t_bad).mean().item()

    print("-" * 40)
    print(f"正常形状得分: {score_real:.2f}")
    print(f"畸形形状得分: {score_bad:.2f}")
    print(f"分差 (Real - Bad): {score_real - score_bad:.2f}")

    if score_real > score_bad:
        print("✅ 判别器初步具备形状先验能力。")
    else:
        print("❌ 判别器目前无法识别畸形。")
    
    # 保存对比图
    vutils.save_image(torch.cat([t_real.cpu(), t_bad.cpu()], 0), 
                      os.path.join(output_dir, "comparison.png"), normalize=True)

if __name__ == "__main__":
    # 建议测试你目前最新的模型（比如 2000, 3000 轮）
    # 注意路径要写对，之前你报错是因为漏了最前面的 / 
    PATH = "/root/example/airs1/semi/code/pretrain/GAN/netD_epoch_8000.pth"
    test_discriminator(PATH)