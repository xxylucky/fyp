from .tn3k import tn3kDataSet
from .BUSI import BUSIDataSet

def build_dataset(args):
    if args.manner == 'test':
        if args.dataset == 'tn3k':
            test_data = tn3kDataSet(args.root, args.expID, mode='test')
        elif args.dataset == 'BUSI':
            test_data = BUSIDataSet(args.root, args.expID, mode='test')  # 原代码args.tn3k,已纠正
        return test_data
    else:  # 'train'
        if args.dataset == 'tn3k':
            train_data = tn3kDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = tn3kDataSet(args.root, args.expID, mode='valid')
            test_data = tn3kDataSet(args.root, args.expID, mode='test')
            train_u_data = None
            if args.manner == 'semi' or args.manner == 'self':
                train_u_data = tn3kDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        elif args.dataset == 'BUSI':
            train_data = BUSIDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='label') #加载有标签训练集，radio=比例
            valid_data = BUSIDataSet(args.root, args.expID, mode='valid')  # 加载验证集
            train_u_data = None  # 初始化无标签训练集为None
            if args.manner == 'semi':  # 如果是半监督学习就需要加载无标签训练集
                train_u_data = BUSIDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        return train_data, train_u_data, valid_data


