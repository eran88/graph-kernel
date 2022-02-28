import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
import torch
random.seed(90)
torch.manual_seed(90)
def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True
    solver = Solver(config)
    solver.train()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=8, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[256,512,1024], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[256, 128], 256, [256, 128]], help='number of conv filters in the first layer of the learned kernels')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in of the learned kernels')
    parser.add_argument('--post_method', type=str, default='soft_gumbel', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])
    parser.add_argument('--random_graph_increase', type=int, default=0, help='increase the nodes representation')


#python main.py --name small_lr  --min_loss 0.5 --change_alpha 100  --batch_size 1024 --g_lr 0.000001  --d_lr 0.000001
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=1024, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.000001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.000001, help='learning rate for D')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--alpha', type=float, default=0.23, help='bandwidth parameter of the GCN kernel')
    parser.add_argument('--laplacian_alpha', type=float, default=1, help='bandwidth parameter of the laplacian kernel')
    parser.add_argument('--shapes_alpha', type=float, default=1, help='bandwidth parameter of the shapes kernel')
    parser.add_argument('--alpha_range', type=int, default=0, help='number of alphas=1+alpha_range*2')
    parser.add_argument('--alpha_mul', type=float , default=2, help='Multiply alpha by')
    parser.add_argument('--num_learned_kernel', type=int, default=1, help='number of learned kernels ')
    parser.add_argument('--num_replacing_kernel', type=int, default=0, help='number of replacing kernels ')
    parser.add_argument('--num_constant_kernel', type=int, default=0, help='number of constant kernels ')
    parser.add_argument('--change_alpha', type=int, default=-1, help='change alpha every x iterations ')
    parser.add_argument('--min_loss', type=float, default=0, help='Train discriminator until he get to -min_loss ')
    parser.add_argument('--laplacian', type=float, default=0.1, help='Laplacian loss alpha ')
    parser.add_argument('--shapes', type=float, default=0.1, help='Shapes loss alpha ')
    parser.add_argument('--learned_alpha', type=str2bool, default=False, help='Set true to learn the bandwidth parameter for the learned kernel ')
    parser.add_argument('--learned_laplacian', type=str2bool, default=True, help='Set true to learn the laplacian kernel ')
    parser.add_argument('--learned_shapes', type=str2bool, default=True, help='Set true to learn the shape kernel ')


    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--samples_iter', type=int, default=100, help='Sample graphs from the generator every x steps')

    # Directories.
    parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes.sparsedataset')
    parser.add_argument('--name', type=str, default='default', help='Location for the models, samples and logs file will be at /results/<name> ')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10, help='output logs after every x steps ')
    parser.add_argument('--model_save_step', type=int, default=10000, help='save the generator after every x steps ')
    parser.add_argument('--lr_update_step', type=int, default=1000, help='decreasing the learning rate after  x steps ')

    config = parser.parse_args()
    print(config)
    main(config)
