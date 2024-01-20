import random
import warnings
from datasets.ferplus_wangkai_dataset import FERPlus_wangkai_dataset
from datasets.raf_with_attri_dataset import RAF_with_attr_dataset
from losses.weighted_asymmetric_loss import WeightedAsymmetricLoss

warnings.filterwarnings("ignore")
from collections import OrderedDict
from sklearn import metrics
from torchinfo import summary
from utils.recoder import *
from losses.focal_loss import FocalLoss2
from models.model_ir50 import Model
import torch.utils.data as data
import os
import argparse
from sklearn.metrics import f1_score, confusion_matrix
from data_preprocessing.sam import SAM
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from config.raf_config import *

warnings.filterwarnings("ignore", category=UserWarning)

#test git

def setup_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


random_seed = 2048
setup_seed(random_seed)

print('model_configs: ', model_configs)
for k in args.__dict__:
    print(k + ": " + str(args.__dict__[k]))
print('random_seed: ', random_seed)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.empty_cache()
    best_acc = 0
    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    # create model
    model = Model(num_classes=args.num_classes).cuda()

    # train recon
    if args.recon:
        checkpoint = torch.load('/home/panr/all_results/POSTER_V2/checkpoint/[06-02]-[22-10]-model_state_dict_best.pth')
        checkpoint.pop('module.vit_head.head.linear.weight')
        new_ckpt = OrderedDict()
        for name, param in checkpoint.items():
            new_name = name.split('module.')[1]
            new_ckpt[new_name] = param
        model.load_state_dict(new_ckpt, strict=False)
    summary(model, input_data=[torch.Tensor(args.batch_size, 3, args.img_size, args.img_size), ])
    # torch.Tensor([0.] * args.batch_size)])

    model = torch.nn.DataParallel(model).cuda()

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = FocalLoss2()
    if args.optimizer == 'adamw':
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")

    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05,
                    adaptive=False, )

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    new_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(f'{len(new_params)} unfreezed layers.')

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    recorder = RecorderMeter(args.epochs)
    recorder1 = RecorderMeter1(args.epochs)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            recorder1 = checkpoint['recorder1']
            best_acc = best_acc.to()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    if args.data_type == 'RAF-DB':
        data_root = '/home/panr/all_datasets/RAF_DB/basic/Partition'
    elif args.data_type == 'FERPlus':
        data_root = '/home/panr/all_datasets/ferplus/Partition'
    elif args.data_type == 'FERPlusCNTK':
        data_root = '/home/panr/all_datasets/ferplus/Partition_CNTK'
    elif args.data_type == 'AffectNet-7':
        data_root = '/home/panr/all_datasets/AffectNet/Partition'
    elif args.data_type == 'AffectNet-7-small':
        data_root = '/home/panr/all_datasets/AffectNet/Partition_small'
    else:
        data_root = '/home/panr/all_datasets/ExpW/ExpW_image_align_filtrate/Partition'
    traindir = os.path.join(data_root, 'train')
    valdir = os.path.join(data_root, 'test')

    if args.data_type == 'RAF-DB':
        data_root = '/home/panr/all_datasets/RAF-DB/basic'
        train_dataset = RAF_with_attr_dataset(root=data_root,
                                              transform=transforms.Compose(
                                                  [transforms.Resize((args.img_size, args.img_size)),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225]),
                                                   # transforms.RandomErasing(scale=(0.02, 0.1))]),
                                                   transforms.RandomErasing()]),
                                              train=True)
        test_dataset = RAF_with_attr_dataset(root=data_root,
                                             transform=transforms.Compose(
                                                 [transforms.Resize((args.img_size, args.img_size)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225]),
                                                  ]),
                                             train=False)
    elif args.data_type == 'FERPlus_wk':
        train_dataset = FERPlus_wangkai_dataset(root='/home/panr/all_datasets/ferplus/',
                                                transform=transforms.Compose(
                                                    [transforms.Resize((args.img_size, args.img_size)),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225]),
                                                     # transforms.RandomErasing(scale=(0.02, 0.1))]),
                                                     transforms.RandomErasing()]),
                                                train=True)
        test_dataset = FERPlus_wangkai_dataset(root='/home/panr/all_datasets/ferplus/',
                                               transform=transforms.Compose(
                                                   [transforms.Resize((args.img_size, args.img_size)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225]),
                                                    ]),
                                               train=False)

    else:
        print('datasets do not exist!')
        return

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if args.evaluate is not None:
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}'".format(args.evaluate))
            checkpoint = torch.load(args.evaluate)
            model.load_state_dict(checkpoint)  # ['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, 1))  # checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.evaluate))
        validate(val_loader, model, criterion,
                 args)  # , 200, anchors, self_attn, loss_fn_mu, loss_fn_center)
        return

    matrix = None
    txt_name = result_path + '/log/' + time_str + 'log.txt'
    with open(txt_name, 'a') as f:
        f.write('Model configs: ' + model_configs + '\n')
        f.write('GPU: ' + args.gpu + '\n')
        f.write('random_seed: ' + str(random_seed) + '\n')
        f.write('load data from: ' + data_root + '\n')

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print('\n\n===================================================================================================')
        print('Current learning rate: ', current_learning_rate)
        with open(txt_name, 'a') as f:
            f.write('\nCurrent learning rate: ' + str(current_learning_rate) + '\n')

        # train for one epoch
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch,
                                     args)

        # evaluate on validation set
        val_acc, val_los, output, target, D = validate(val_loader, model, criterion,
                                                       args)

        scheduler.step()
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder1.update(output, target)

        curve_name = time_str + 'cnn.png'
        recorder.plot_curve(os.path.join(result_path + '/log/', curve_name))
        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        print('Current best accuracy: ', best_acc.item())
        if is_best:
            matrix = D
        print('Current best matrix: \n', matrix)

        txt_name = result_path + '/log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder1': recorder1,
                         'recorder': recorder}, is_best, args)
    return np.round(best_acc.item(), 2)


def train(train_loader, model, criterion, optimizer, epoch,
          args):  # , anchors, self_attn, loss_fn_mu, loss_fn_center):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, labels.long())  # + 0.5 * partition_loss
        # measure accuracy and record loss
        acc1, _ = accuracy(output, labels, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.first_step()

        # compute output
        output = model(images)
        loss = criterion(output, labels.long())  # + 0.5 * partition_loss
        # measure accuracy and record loss
        acc1, _ = accuracy(output, labels, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.second_step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i, result_path, time_str)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion,
             args):  # , epoch, anchors, self_attn, loss_fn_mu, loss_fn_center):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    D = [[0] * args.num_classes] * args.num_classes
    Labels = [i for i in range(args.num_classes)]

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()
            output = model(images)  # clt_hat:1,7,786  delta_l2:b,7
            loss = criterion(output, labels.long())  # + 0.5 * partition_loss

            # measure accuracy and record loss
            acc, _ = accuracy(output, labels, topk=(1, 2))
            losses.update(loss.item(), images.size(0))
            top1.update(acc[0], images.size(0))
            topk = (1,)
            # Computes the accuracy over the k top predictions for the specified values of k
            with torch.no_grad():
                maxk = max(topk)
                # batch_size = labels.size(0)
                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()

            output = pred
            labels = labels.squeeze().cpu().numpy()
            output = output.squeeze().cpu().numpy()
            im_re_label = np.array(labels)
            im_pre_label = np.array(output)
            y_ture = im_re_label.flatten()
            im_re_label.transpose()
            y_pred = im_pre_label.flatten()
            im_pre_label.transpose()

            C = metrics.confusion_matrix(y_ture, y_pred, labels=Labels)
            D += C
            if i % (val_loader.__len__() // 2) == 0:
                progress.display(i, result_path, time_str)
        print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        with open(result_path + '/log/' + time_str + 'log.txt', 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
    print(D)

    return top1.avg, losses.avg, output, labels, D


def save_checkpoint(state, is_best, args):
    # torch.save(state, args.checkpoint_path)
    torch.save(state, args.checkpoint_path[:-4] + '_state_dict.pth')
    if is_best:
        best_state = state.pop('optimizer')
        # torch.save(best_state, args.best_checkpoint_path)
        torch.save(state, args.checkpoint_path[:-4] + '_state_dict_best.pth')


if __name__ == '__main__':
    best_acc = main()
    end_time = datetime.datetime.now()
    end_time_str = end_time.strftime("[%m-%d]-[%H-%M]")
    f = open(result_path + f'/log_run/{model_configs}.log', 'a')
    f.write(f'\nEnd_time: {end_time_str}')
    f.close()
    os.rename(result_path + f'/log_run/{model_configs}.log',
              result_path + f'/log_run/{time_str}{best_acc}_{model_configs}.log')
