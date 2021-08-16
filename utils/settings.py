import argparse
import time
import torch.nn as nn
import torch

def parse_opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')

    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size')
    parser.add_argument('--iters', type = int, default=3, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='fedbn', help='[FedBN | FedAvg | FedProx]')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint/domainnet/', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--group', action='store_true', help='use group norm')
    parser.add_argument('--num_groups', default=2, help='numbers of groups for group norm')
    parser.add_argument('--conv_only', action='store_true', help='average convolutional layer only')
    parser.add_argument('--pool', action='store_true', help='pool all data to train a centerlized model')
    args = parser.parse_args()
    
    args.datasets = ['Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch']
    args.client_num = len(args.datasets)
    # saved info
    if args.pool:
        dir_path = "../log/pool/"
        args.save_path += "pool/"
        fig_path = "../figs/pool/"
    else:
        dir_path = "../log/fed/"
        args.save_path += "fed/"
        fig_path = "../figs/fed/"

    if args.group:
        args.log_path = dir_path + 'gn_{}_{}_{}.txt'.format(args.num_groups, args.conv_only, args.mode)
        args.model_path = args.save_path + 'gn_{}_{}_{}.pt'.format(args.num_groups, args.conv_only, args.mode)
        args.fig_path = fig_path + 'gn_{}_{}_{}'.format(args.num_groups, args.conv_only, args.mode)
    else:
        args.log_path = dir_path + 'bn_{}_{}.txt'.format(args.conv_only, args.mode)
        args.model_path = args.save_path + 'bn_{}_{}.pt'.format(args.conv_only, args.mode)
        args.fig_path = fig_path + 'bn_{}_{}'.format(args.conv_only, args.mode)

    # setttings for model training
    args.loss_func = nn.CrossEntropyLoss()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.use_gpu = torch.cuda.is_available()


    return args


def write_log_head(logfile, args):
    logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    logfile.write('===Setting===\n')
    logfile.write('    lr: {}\n'.format(args.lr))
    logfile.write('    batch: {}\n'.format(args.batch))
    logfile.write('    iters: {}\n'.format(args.iters))
    logfile.write('    use group norm: {}\n'.format(args.group))
    logfile.write('    group number: {}\n'.format(args.num_groups))

def write_log_body(logfile, log_str, epoch, total_epoch):
    logfile.write("============ Train epoch {}/{} ============\n".format(epoch, total_epoch))
    logfile.write(log_str)

# test case
if __name__ == "__main__":
    opt = parse_opts()
    print(opt)