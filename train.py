from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC',
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--resume_weights_only', default=False, type=str2bool,
                    help='Arguments to resume only weights (not epoch, lr, etc)')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='Resume training at this epoch')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# Make the default save_folder a subdirectory of the script folder.
script_path = os.path.dirname(os.path.realpath(sys.argv[0])) + '/'
if args.save_folder == parser.get_default('save_folder'):
    args.save_folder = os.path.join(script_path, args.save_folder)

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        dataset_config = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(dataset_config['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        dataset_config = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(dataset_config['min_dim'],
                                                         MEANS))
    elif args.dataset in ['Tree28_synthesis1', 'Tree29_synthesis1']:
        dataset_config = tree_synth0_config
        dataset = TreeDataset(root=args.dataset_root, name=args.dataset,
                           transform=SSDAugmentation(dataset_config['min_dim'],
                                                     dataset_config['pixel_means']))
    elif args.dataset in ['Tree28_synthesis2', 'Tree29_synthesis2']:
        dataset_config = tree_synth1_config
        dataset = TreeDataset(root=args.dataset_root, name=args.dataset,
                           transform=SSDAugmentation(dataset_config['min_dim'],
                                                     dataset_config['pixel_means']))
    elif args.dataset in ['Tree28_synthesis3', 'Tree29_synthesis3', 'Tree30_synthesis4']:
        dataset_config = tree_synth2_config
        dataset = TreeDataset(root=args.dataset_root, name=args.dataset,
                          transform=SSDAugmentation(dataset_config['min_dim'],
                                                    dataset_config['pixel_means']))
    else:
        raise ValueError('The dataset is not defined.')

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    # Initialize net
    net = build_ssd('train', dataset_config)

    # Load weights.
    if args.cuda:
        Map_loc = lambda storage, loc: storage
    else:
        Map_loc='cpu'

    if args.resume:
        print('Resuming training. Loading {}...'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=Map_loc)
        if 'net_state' in checkpoint.keys():
            net.load_state_dict(checkpoint['net_state'])
            if not args.resume_weights_only:
                args.start_epoch = checkpoint['epoch']
                print('Adjusting the learning rate to: {}'.format(checkpoint['lr']))
                args.lr = checkpoint['lr']
                # optimizer.load_state_dict(checkpoint['optimizer_state'])
        else:
            net.load_weights(args.resume)
    else:
        print('Loading base network...')
        vgg_weights = torch.load(args.save_folder + args.basenet, map_location=Map_loc)
        net.vgg.load_state_dict(vgg_weights)

        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        net.extras.apply(weights_init)
        net.loc.apply(weights_init)
        net.conf.apply(weights_init)

    if args.cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net = net.cuda()

    # Initialize optimizer and criterion.
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(dataset_config, 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    net.train()
    print('Training SSD on:', dataset.name, 'for {} epochs.'.format(dataset_config['N_epochs']))
    print('Using the following args:')
    print(args)

    N_iterations = len(dataset) // args.batch_size

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    for epoch in range(args.start_epoch, dataset_config['N_epochs']):
        if args.visdom and epoch != 0:
            update_vis_plot(epoch, epoch_loc_loss, epoch_conf_loss, epoch_plot, None,
                            'append', N_iterations)

        # reset epoch losses
        epoch_loc_loss = 0
        epoch_conf_loss = 0
        epoch_total_loss = 0
        epoch_avg_loss = 0

        if epoch in dataset_config['lr_steps']:
            adjust_learning_rate(optimizer, args.gamma)

        # loop through all batches
        t0 = time.time()
        for iteration, Data in enumerate(data_loader):
            images, targets = Data
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]
            # forward prop
            out = net(images)

            # backward prop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()

            # save epoch losses
            epoch_loc_loss += loss_l.data[0]
            epoch_conf_loss += loss_c.data[0]
            epoch_total_loss = epoch_loc_loss + epoch_conf_loss
            epoch_avg_loss = epoch_total_loss/((iteration+1) * args.batch_size)

            # monitoring
            if iteration % 10 == 0:
                t1 = time.time()
                print("Iteration {:4d} || Epoch Avg Loss {:.4f} || timer: {:.2f} s".format(iteration, epoch_avg_loss, (t1 - t0)))
                t0 = time.time()

            if args.visdom:
                update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                                iter_plot, epoch_plot, 'append')

        # save checkpoint.
        if epoch != 0 and epoch % 2 == 0:
            print('Saving checkpoint, epoch:', epoch)
            checkpoint_filename = args.save_folder + 'ssd300_' + args.dataset + '_' + repr(epoch) + '.pth'
            save_checkpoint(net, args.lr, epoch, epoch_loc_loss, epoch_conf_loss,
                            epoch_total_loss, epoch_avg_loss, checkpoint_filename)

    # save final state.
    checkpoint_filename = args.save_folder + 'ssd300_' + args.dataset + '_Final.pth'
    save_checkpoint(net, args.lr, epoch, epoch_loc_loss, epoch_conf_loss,
                    epoch_total_loss, epoch_avg_loss, checkpoint_filename)
    torch.save(net.state_dict(), args.save_folder + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    args.lr = args.lr * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def save_checkpoint(net, lr, epoch, epoch_loc_loss, epoch_conf_loss, epoch_total_loss, epoch_avg_loss, filename):
    checkpoint_dict = {'epoch': epoch + 1,
                       'net_state': net.state_dict(),
                       'lr': lr,
                       # 'optimizer_state': optimizer.state_dict(),
                       'loc_loss': epoch_loc_loss,
                       'conf_loss': epoch_conf_loss,
                       'total_loss': epoch_total_loss,
                       'avg_loss': epoch_avg_loss}
    torch.save(checkpoint_dict, filename)

def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
