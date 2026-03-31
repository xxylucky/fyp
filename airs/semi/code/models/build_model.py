import torch
import models
import os

def build_model(args):
    model = getattr(models, args.model)(args.nclasses, args.band)
    # getattr(models, args.model) 的意思是：
    # 从 models 文件夹里找名字叫 args.model (比如 "UNet") 的类。
    # 然后传入 (类别数, 通道数) 进行实例化。
    if args.GPUs:
        model.cuda()
        torch.backends.cudnn.benchmark = True
    if args.load_ckpt is not None:
        model_dict = model.state_dict()

        load_ckpt_path = os.path.join(args.root, "semi/checkpoint/" + str(args.ckpt_name), args.load_ckpt + '.pth')
        print(load_ckpt_path)  # 拼接存放训练好参数的文件路径 (.pth 文件)

        assert os.path.isfile(load_ckpt_path), 'No checkpoint found.'  # 检查文件是否存在，不存在就报错（类似 C++ 的 assert）
        
        print('Loading checkpoint......')
        checkpoint = torch.load(load_ckpt_path)

        # --- 关键逻辑：参数匹配 ---
        # 这一行叫字典推导式。它的作用是：
        # 如果硬盘文件里的参数名(k)在当前模型里能对上，就把它挑出来。
        # 这可以防止因为模型微调导致参数名不匹配而崩溃。
        new_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}

        # 更新参数并装载回模型
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        print('Done')

    return model

# 这样做的好处是：
# 如果你以后在 models 文件夹下加了一个新模型叫 MyNewNet，
# 你只需要在命令行输入 --model MyNewNet 即可，不需要修改这一行代码。
