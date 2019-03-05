'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse
import os
import random
import shutil
import time
import warnings
import sys

from models import *
#from utils import progress_bar
from logger import Logger


logger = Logger('./logs')

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--ckpath', default="ckpt",help="the name of checkpoint")
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoches', default=300, type=float, help='number of epoches')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true', help='testing otherwise training')
parser.add_argument('--tofile', '-f', action='store_true', help='store to file')
parser.add_argument('--cut', '-c', action='store_true', help='NAS15_CUT')
args = parser.parse_args()


if args.tofile:
	saveout = sys.stdout
	saveerr = sys.stderr
	fsock = open("logs/"+str(args.ckpath)+'.log', 'w+')
	esock = open("logs/"+str(args.ckpath)+'-err.log', 'w+')
	sys.stdout = fsock
	sys.stderr = esock

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    initial_lr = args.lr
    if epoch <= 150:
        lr = initial_lr
    elif epoch <=225:
        lr = initial_lr/10
    else:
        lr = initial_lr/100

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("="*100)
    print('At epoch:',epoch," lr is:",lr)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.cut:
    net = NAS15_cut()
else:
    net = NAS15()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..','./checkpoint/'+str(args.ckpath)+".t7")
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+str(args.ckpath)+".t7")
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("Best Acc:", best_acc, " Start Epoch:",start_epoch)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4,nesterov=True)





class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Training
def train(epoch):
    # print('Epoch: %d' % epoch)    
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)


        inputs, targets = inputs.to(device), targets.to(device)        
        
        # compute output
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))        
        top5.update(acc5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % (len(trainloader)//9) == 0 or batch_idx+1==len(trainloader):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.sum:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, batch_idx, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    with torch.no_grad():
        end = time.time()
        for batch_idx, (inputs, targets) in enumerate(testloader):


            inputs, targets = inputs.to(device), targets.to(device)
            
            # compute output
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if batch_idx % (len(testloader)//2) == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       batch_idx, len(testloader), batch_time=batch_time, loss=losses,top1=top1))

        print(' * Acc@1 {top1.avg:.3f} Previous Best {best_acc}'.format(top1=top1,best_acc=best_acc))

        # return top1.avg
        #     test_loss += loss.item()
        #     _, predicted = outputs.max(1)
        #     total += targets.size(0)
        #     correct += predicted.eq(targets).sum().item()

        #     progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = top1.avg
    if acc > best_acc and not args.test:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+str(args.ckpath)+'.t7')
        best_acc = acc
    return best_acc

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



if args.test:
	final_best = test(0)
else:
	for epoch in range(start_epoch, start_epoch+int(args.epoches)):    
	    adjust_learning_rate(optimizer,epoch)
	    train(epoch)
	    final_best = test(epoch)

    
  
print(final_best)

if args.tofile:
	sys.stdout = saveout
	sys.stderr = saveerr
	fsock.close()
	esock.close()
