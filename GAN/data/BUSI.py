from torch.utils.data import Dataset
from torchvision import transforms
import os
from . utils.mytransforms import *
import json


class BUSIDataSet(Dataset):
    def __init__(self, root, expID, mode='train', ratio=10, sign='label', transform=None):
        super(BUSIDataSet, self).__init__()
        self.mode = mode
        self.sign = sign
        
        # 1. 确定文件列表 (逻辑不变)
        if mode == 'train':
            if sign =='label':
                if expID == 1:
                    imgfile = os.path.join(root,'data/splits/BUSI/72/labeled.txt')
                elif expID == 2:
                    imgfile = os.path.join(root,'data/splits/BUSI/144/labeled.txt')
                elif expID == 3:
                    imgfile = os.path.join(root,'data/splits/BUSI/288/labeled.txt')
                with open(imgfile,'r') as f:
                    imglist = f.read().splitlines()
                    self.imglist = [os.path.join(root, 'data/BUSI',img) for img in imglist]
            else:
                if expID == 1:
                    imgfile = os.path.join(root,'data/splits/BUSI/72/unlabeled.txt')
                elif expID == 2:
                    imgfile = os.path.join(root,'data/splits/BUSI/144/unlabeled.txt')
                elif expID == 3:
                    imgfile = os.path.join(root,'data/splits/BUSI/288/unlabeled.txt')
                with open(imgfile,'r') as f:
                    imglist = f.read().splitlines()
                    self.imglist = [os.path.join(root, 'data/BUSI',img) for img in imglist]
        elif mode == 'valid':
            imgfile = os.path.join(root,'data/splits/BUSI/val.txt')
            with open(imgfile,'r') as f:
                    imglist = f.read().splitlines()
                    self.imglist = [os.path.join(root, 'data/BUSI',img) for img in imglist]
        elif mode == 'test':
            imgfile = os.path.join(root, 'data/BUSI/leftImg/val')
            self.imglist = [file for file in self.get_all_files(imgfile)]

        # 2. 预加载数据到内存
        self.data_cache = []
        print(f"正在预加载 {self.mode}-{self.sign} 数据 (共 {len(self.imglist)} 张)...")
        
        for img_path in self.imglist:
            img = Image.open(img_path).convert('RGB')
            
            if self.mode == 'train' and self.sign == 'unlabel':
                # --- 关键修改：无标签也造一个全黑的虚假标签，为了兼容 mytransforms ---
                fake_gt = Image.new('L', img.size, 0) 
                self.data_cache.append({'image': img, 'label': fake_gt})
            else:
                gt_path = img_path.replace('leftImg', 'gtFine')
                gt = Image.open(gt_path).convert('L')
                self.data_cache.append({
                    'image': img, 
                    'label': gt, 
                    'name': img_path.split('/')[-1]
                })
        print(f"✅ 数据已锁入内存！")

        # 在 BUSI.py 的 __init__ 中，将原来的 Compose 替换为：
        if transform is None:
            if mode == 'train' and sign == 'label':
                transform = transforms.Compose([
                    Resize((64, 64)),  # <--- 关键：直接在这里统一到 64x64
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    # 注意：如果 RandomZoom 还是很慢，可以暂时注释掉它
                    # RandomZoom((0.9, 1.1)), 
                    ToTensor(),
                    # Normalize((0.5,), (0.5,))  # <--- 添加这一行，将 [0,1] 变为 [-1,1]
                ])
            elif mode == 'train' and sign == 'unlabel':
                transform = transforms.Compose([
                    Resize((64, 64)),  # <--- 直接统一到 64x64
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    ToTensor(),
                    # Normalize((0.5,), (0.5,))  # <--- 添加这一行，将 [0,1] 变为 [-1,1]
                ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                   Resize((64, 64)),
                   ToTensor(),
                   # Normalize((0.5,), (0.5,))  # <--- 添加这一行，将 [0,1] 变为 [-1,1]
                ])
        self.transform = transform
    
    def __getitem__(self, index):
        item = self.data_cache[index]
        
        # 无论什么模式，先复制 image 和 label 以供 transform 使用
        data = {
            'image': item['image'],
            'label': item['label']
        }
        
        if self.transform:
            data = self.transform(data)
            
        if self.mode == 'train' and self.sign == 'unlabel':
            # GAN 模式：返回单个 Tensor
            return data['image'] 
        else:
            # 监督学习模式：返回带名字的字典
            # 安全地获取 name，防止 unlabel 模式误入此处报错
            data['name'] = item.get('name', 'unknown') 
            return data

    def __len__(self):
        # 既然数据在内存，直接返回缓存长度最准
        return len(self.data_cache)
    
    
    
    def __len__(self):
        return len(self.imglist)
    
    def get_all_files(self,directory):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(file_path)
        return file_paths
