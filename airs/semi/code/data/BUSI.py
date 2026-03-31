from torch.utils.data import Dataset
from torchvision import transforms
import os
from utils.mytransforms import *
import json


class BUSIDataSet(Dataset):
    def __init__(self, root, expID, mode='train', ratio=10, sign='label', transform=None):
        super(BUSIDataSet, self).__init__()  # 父类继承，传入mode和sign
        self.mode = mode
        self.sign = sign

        # 传入数据
        if mode == 'train':  # 训练模式
            if sign =='label':  # 加载有标签数据，根据expID选择对应的数据量大小，expID定义在opt.py
                if expID == 1:
                    imgfile = os.path.join(root,'data/splits/BUSI/72/labeled.txt')
                elif expID == 2:
                    imgfile = os.path.join(root,'data/splits/BUSI/144/labeled.txt')
                elif expID == 3:
                    imgfile = os.path.join(root,'data/splits/BUSI/288/labeled.txt')
                with open(imgfile,'r') as f: 
                    imglist = f.read().splitlines() # 把 txt 文件读进来，按行切分成一个列表（Array）
                    # 将 root 路径、数据集文件夹名和文件名拼接成完整路径
                    self.imglist = [os.path.join(root, 'data/BUSI',img) for img in imglist] 
            else:  # 加载无标签数据，逻辑同上，区别在于读取unlabeled.txt文件
                if expID == 1:
                    imgfile = os.path.join(root,'data/splits/BUSI/72/unlabeled.txt')
                elif expID == 2:
                    imgfile = os.path.join(root,'data/splits/BUSI/144/unlabeled.txt')
                elif expID == 3:
                    imgfile = os.path.join(root,'data/splits/BUSI/288/unlabeled.txt')
                with open(imgfile,'r') as f:
                    imglist = f.read().splitlines()
                    self.imglist = [os.path.join(root, 'data/BUSI',img) for img in imglist]
        elif mode == 'valid':  # 验证模式，加载验证集val
            imgfile = os.path.join(root,'data/splits/BUSI/val.txt')
            with open(imgfile,'r') as f:
                    imglist = f.read().splitlines()
                    self.imglist = [os.path.join(root, 'data/BUSI',img) for img in imglist]
        elif mode == 'test':  # 测试模式，
            imgfile = os.path.join(root, 'data/BUSI/leftImg/val')
            self.imglist = [file for file in self.get_all_files(imgfile)]  # 调用类内函数 get_all_files 获取该文件夹下所有图片的绝对路径

        # 数据预处理流程设计：进行随机变换。是防止过拟合的最有效手段
        if transform is None:  # 默认变换方式
            if mode == 'train' and sign == 'label':  # 有标签
               transform = transforms.Compose([
                   Resize((320, 320)),         # 先统一缩放到 320x320
                   RandomHorizontalFlip(),     # 随机左右翻转
                   RandomVerticalFlip(),       # 随机上下翻转
                   RandomRotation(90),         # 随机旋转（最大90度）
                   RandomZoom((0.9, 1.1)),     # 随机缩放（0.9倍到1.1倍之间）
                   RandomCrop((256, 256)),     # 从中心或随机位置裁剪出 256x256
                   ToTensor()                  # 核心：将图像转为数学矩阵(Tensor)
               ])
            elif mode == 'train' and sign == 'unlabel':  # 无标签，少了随机缩放0.9-1.1
                transform = transforms.Compose([
                    transforms.Resize((320, 320)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90),
                    transforms.RandomCrop((256, 256)),
                    transforms.ToTensor()
                ])
            elif mode == 'valid' or mode == 'test':  # 验证集和测试集，不用变换
                transform = transforms.Compose([
                   Resize((320, 320)),
                   ToTensor()
                ])
            # label 部分（没有前缀）：使用的是 from utils.mytransforms import *。
            #       因为用了 *，所以 Resize、RandomRotation 这些函数被直接“搬”到了当前文件的命名空间里。你直接喊名字，程序就能识别。
            # unlabel 部分（有前缀）：使用的是 transforms.Resize。这调用的是官方 torchvision 库里的标准函数。
        self.transform = transform  # 将变换流程保存在transform中，共后续使用

    def __getitem__(self, index):  # 根据一个索引（index），返回一组可用于训练的数据
        if self.mode == 'train' and self.sign == 'unlabel':
            img_path = self.imglist[index]  # 根据index找到图片路径
            img = Image.open(img_path).convert('RGB')  # 打开图片并转为RGB三通道矩阵
            if self.transform:
                return self.transform(img)  # 返回处理好的Tensor矩阵
        else:  # 有标签、验证集、测试集
            img_path = self.imglist[index]  # 根据index找到图片路径
            # 通过替换字符串，从原图路径推导出对应的标签（Ground Truth）路径
            # 假设原图在 leftImg 文件夹，对应的黑白标签就在 gtFine 文件夹
            gt_path = img_path.replace('leftImg', 'gtFine')  
            img = Image.open(img_path).convert('RGB')  # 原图转RGB
            gt = Image.open(gt_path).convert('L')  # 标签图转灰度
            data = {'image': img, 'label': gt}  # 将图和标签打包成一个字典 (Dictionary)，类似 C++ 的 struct
            if self.transform:
                # ！！！注意：这里的 transform 会同时作用于图像和标签，确保变换一致
                data = self.transform(data)
            data['name'] = self.imglist[index].split('/')[-1]  # split('/')[-1] 表示按斜杠切分后取最后一段
            return data  # 返回包含原图矩阵、标签矩阵和文件名的dict

    def __len__(self):
        return len(self.imglist)  # 返回存储路径的列表长度，告诉模型一共有多少数据
        # steps = __len__ / batch_size ，in one epoch
    
    # 这个函数只在 mode == 'test' 时被调用。
    def get_all_files(self,directory):
        file_paths = []  # 初始化一个空列表，用来存所有发现的文件路径
        for root, dirs, files in os.walk(directory):  # os.walk 递归遍历文件夹
            for file in files:
                file_path = os.path.abspath(os.path.join(root, file))  # os.path.abspath()将括号内转为绝对路径
                file_paths.append(file_path)  # 将路径加入列表
        return file_paths
