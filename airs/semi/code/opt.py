import argparse

parse = argparse.ArgumentParser(description='PyTorch Semi-Medical-Seg Implement')

"-------------------GPU option----------------------------"
parse.add_argument('--GPUs', type=str, default='0')

"-------------------data option--------------------------"
parse.add_argument('--root', type=str, default='/root/example/fyp/airs')  # 需要修改default为不同的路径以实现不同数据集的训练和测试
parse.add_argument('--dataset', type=str, default='BUSI', choices=['tn3k','BUSI'])  # 选择tn3k或BUSI数据集进行训练和测试
parse.add_argument('--ratio', type=int, default=10)  # 需要修改default为不同的值以实现不同的标签比例（如10表示10%标签数据，90%未标签数据）

"-------------------training option-----------------------"
parse.add_argument('--manner', type=str, default='semi', choices=['full', 'semi', 'test', 'self'])  # 需要修改default为不同的模式以实现模型训练和测试
parse.add_argument('--mode', type=str, default='train')
parse.add_argument('--nEpoch', type=int, default=200)
parse.add_argument('--batch_size', type=int, default=24)
parse.add_argument('--num_workers', type=int, default=2)
parse.add_argument('--load_ckpt', type=str, default=None)  # test时可改为best、second_best、third_best等以加载不同的模型权重进行测试
parse.add_argument('--model', type=str, default='MyModel', choices=['MyModel', 'MyModel_aspp'])  # 选用ASPP
parse.add_argument('--expID', type=int, default=1)  # 需要修改default：1--72 2--144 3--288
parse.add_argument('--ckpt_name', type=str, default='default_exp')  # 指定模型训练过程中保存权重（checkpoint）文件夹的名字。它通常用于区分不同实验或参数设置下的模型结果。
parse.add_argument('--samAfter', type=int, default=30)  # 需要修改default为不同的值以实现不同的训练轮数后接入SAM

"-------------------optimizer option-----------------------"
parse.add_argument('--lr', type=float, default=1e-3)
parse.add_argument('--power',type=float, default=0.9)
parse.add_argument('--betas', default=(0.9, 0.999))
parse.add_argument('--weight_decay', type=float, default=1e-5)
parse.add_argument('--eps', type=float, default=1e-8)
parse.add_argument('--mt', type=float, default=0.9)
parse.add_argument('--nclasses', type=int, default=1)
parse.add_argument('--band', type=int, default=3)

"-------------------topology / betti matching option-----------------------"
# 是否启用监督分支拓扑损失：mask vs gt
parse.add_argument('--use_topo_sup', type=int, default=1,
                   help='1: enable topology loss on supervised branch (mask vs gt), 0: disable')

# 是否启用半监督分支拓扑损失：predboud vs pseudo
parse.add_argument('--use_topo_semi', type=int, default=1,
                   help='1: enable topology loss on semi branch (predboud vs pseudo), 0: disable')

# 控制半监督分支从第几个 epoch 开始引入 topo
# 例如：
# 0  -> 从一开始就加 semi topo
# 30 -> 前30个epoch只加监督分支 topo，第30个epoch开始加 semi topo
parse.add_argument('--use_semi', type=int, default=30,
                   help='epoch to start topology loss on semi branch')

# 监督分支 topo loss 权重
parse.add_argument('--lambda_topo_sup', type=float, default=1.0,
                   help='weight of topology loss on supervised branch')

# 半监督分支 topo loss 权重
parse.add_argument('--lambda_topo_semi', type=float, default=1.0,
                   help='weight of topology loss on semi branch')

# matched / unmatched 内部权重
parse.add_argument('--lambda_topo_match', type=float, default=1.0,
                   help='weight of matched term inside topology loss')
parse.add_argument('--lambda_topo_unmatch', type=float, default=1.0,
                   help='weight of unmatched term inside topology loss')

# 只做哪一维拓扑：0=components, 1=holes
# 对乳腺病灶分割建议先用 0
parse.add_argument('--topo_hdim', type=int, default=0, choices=[0, 1],
                   help='homology dimension for topology loss: 0 for connected components, 1 for holes')

# backend 调用时是否计算 target 侧 unmatched
# 训练时通常不需要，关掉更省
parse.add_argument('--topo_include_target_unmatched', type=int, default=0,
                   help='whether to include unmatched pairs on target side')

# 对 prediction 做 topology loss 时是否先 threshold 成二值
# strict Betti matching 一般仍建议输入概率图，不建议这里硬阈值
parse.add_argument('--topo_use_prob', type=int, default=1,
                   help='1: use probability map for topology loss, 0: binarize before topology loss')

# 防止 backend 没有 interval 时除0或返回空
parse.add_argument('--topo_eps', type=float, default=1e-8,
                   help='numerical stability for topology loss')

# 如果你后续想加伪标签置信筛选，可以先预留
parse.add_argument('--topo_conf_thresh', type=float, default=0.0,
                   help='confidence threshold for semi topology loss; 0 means disabled')

parse.add_argument('--topo_backend_root', type=str,
                   default='/root/example/fyp/airs/Betti-Matching-3D-master')
parse.add_argument('--topo_backend_import', type=str,
                   default='build.betti_matching')

args = parse.parse_args()
