import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--imageSize', type=tuple, default=256)#256*512
parser.add_argument('--last_ckpt', type=str, default="logs/checkpoint/swin_tiny_18.668222_4.pth", help='continue the training process')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--n_knn', type=int, default=100, help='num of knn neighbors')
parser.add_argument('--train_path', default='trainset', help='folder to train data')
parser.add_argument('--valid_path', default='validset', help='folder to train data')
parser.add_argument('--test_path', default='test_images_128x128', help='folder to valid data')
parser.add_argument('--resume', default=False, help='folder to output images and model checkpoints')
opt = parser.parse_args()