import argparse

parse = argparse.ArgumentParser(description='PyTorch Semi-Medical-Seg Implement')

"-------------------GPU option----------------------------"
parse.add_argument('--GPUs', type=str, default='0')

"-------------------data option--------------------------"
parse.add_argument('--root', type=str, default='/root/example/airs1')  # 需要修改default为不同的路径以实现不同数据集的训练和测试
parse.add_argument('--dataset', type=str, default='BUSI', choices=['tn3k','BUSI'])  # 选择tn3k或BUSI数据集进行训练和测试
parse.add_argument('--ratio', type=int, default=10)  # 需要修改default为不同的值以实现不同的标签比例（如10表示10%标签数据，90%未标签数据）

"-------------------training option-----------------------"
parse.add_argument('--manner', type=str, default='semi', choices=['full', 'semi', 'test', 'self'])  # 需要修改default为不同的模式以实现模型训练和测试
parse.add_argument('--mode', type=str, default='train')
parse.add_argument('--nEpoch', type=int, default=200)
parse.add_argument('--batch_size', type=int, default=24)
parse.add_argument('--num_workers', type=int, default=2)
parse.add_argument('--load_ckpt', type=str, default=None)  # test时可改为best、second_best、third_best等以加载不同的模型权重进行测试
parse.add_argument('--model', type=str, default='MyModel')
parse.add_argument('--expID', type=int, default=1)  # 需要修改default：1--72 2--144 3--288
parse.add_argument('--ckpt_name', type=str, default='default_exp')  # 指定模型训练过程中保存权重（checkpoint）文件夹的名字。它通常用于区分不同实验或参数设置下的模型结果。
parse.add_argument('--samEpoch', type=int, default=30)  # 需要修改default为不同的值以实现不同的训练轮数后接入SAM

"-------------------optimizer option-----------------------"
parse.add_argument('--lr', type=float, default=1e-3)
parse.add_argument('--power',type=float, default=0.9)
parse.add_argument('--betas', default=(0.9, 0.999))
parse.add_argument('--weight_decay', type=float, default=1e-5)
parse.add_argument('--eps', type=float, default=1e-8)
parse.add_argument('--mt', type=float, default=0.9)
parse.add_argument('--nclasses', type=int, default=1)
parse.add_argument('--band', type=int, default=3)

args = parse.parse_args()
