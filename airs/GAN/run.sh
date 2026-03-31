export CUDA_VISIBLE_DEVICES=0
# 指定使用GPU 0进行训练，如果有多块GPU，可以修改为1、2等，或者使用CUDA_VISIBLE_DEVICES=0,1来指定多个GPU。

nohup python -u main.py > train_gan.log --dataset busi --expID 1 --cuda 2>&1 &  
# 使用nohup命令在后台运行训练脚本，并将输出日志保存到train_gan.log文件中。
# --dataset BUSI指定使用BUSI数据集进行训练。
# --expID 1指定实验ID为1，可以根据需要修改为2、3等。
# --cuda表示使用GPU进行训练。
# 训练完成后，可以使用tail -f train_gan.log命令实时查看训练