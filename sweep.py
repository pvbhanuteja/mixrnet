'''Train CIFAR10 with PyTorch.'''
# from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import uuid
import time
import wandb
import os
import argparse
from dataloader import  CIFAR_Dataset
from mixup import mixup_cross_entropy_loss
from models import *
from utils import progress_bar, get_time_str
from sweep_config import sweep_config
os.environ["WANDB_API_KEY"] = '3bd98dfca3ee43d5b0750b7e7ae85ba3a422d0d2'
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--run_name','-rn',type=str, help='your experiment name',default=f'{get_time_str()}_default')
parser.add_argument('--batch_size', '-b', default=128, type=int, help='batch size')
parser.add_argument('--ckp_last', '-cl', default=True, type=bool, help='resume with last checkpoint if false resume with best checkpoint')
parser.add_argument('--num_epochs', '-ne', default=200, type=int, help='number of epochs')


args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)


# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testset = CIFAR_Dataset('data/cifar-10-batches-py/', False, transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet50PreAct()

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

wandb_projname = args.run_name
args.run_name = f'runs/{args.run_name}/{get_time_str()}'

def main():
    with wandb.init() as run:
        trainset = CIFAR_Dataset('data/cifar-10-batches-py/', True, transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        if wandb.config.optimizer == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=wandb.config.lr,
                        momentum=0.9, weight_decay=wandb.config.wd)
        if wandb.config.optimizer == 'adam':
            optimizer = optim.Adam(net.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        for epoch in range(0, 200):
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs = Variable(inputs, requires_grad=True)
                targets = Variable(targets, requires_grad=False)
                inputs, targets = inputs.to(device), targets.to(device)
                # print(inputs.shape,targets.shape)
                optimizer.zero_grad()
                outputs = net(inputs)
                # loss = criterion(outputs, targets)
                loss = mixup_cross_entropy_loss(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                # _, predicted = outputs.max(1)
                predicted = outputs.data.max(1, keepdim=True)[1]
                total += targets.size(0)
                # correct += predicted.eq(targets).sum().item()
                correct += predicted.eq(targets.data.max(1, keepdim=True)[1]).sum()

                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            wandb.log({"train_loss": train_loss/(batch_idx+1),
                        "train_accuracy":100.*correct/total,
                        "epoch":epoch})
            if (epoch+1)%10 == 0:
                print('Saving model..')
                state = {
                    'net': net.state_dict(),
                    'acc': 100.*correct/total,
                    'epoch': epoch,
                }
                if not os.path.isdir(args.run_name):
                    os.makedirs(args.run_name)
                torch.save(state, f'./{args.run_name}/ckpt_last.pth')

            global best_acc
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = mixup_cross_entropy_loss(outputs, targets)

                    test_loss += loss.item()
                    predicted = outputs.data.max(1, keepdim=True)[1]
                    total += targets.size(0)
                    correct += predicted.eq(targets.data.max(1, keepdim=True)[1]).sum()

                    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # Save checkpoint.
            acc = 100.*correct/total
            wandb.log({"test_loss": test_loss/(batch_idx+1),
                    "test_accuracy":acc,
                    "epoch":epoch})
            if acc > best_acc:
                print('Saving best model..')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                if not os.path.isdir(args.run_name):
                    os.makedirs(args.run_name)
                torch.save(state, f'./{args.run_name}/ckpt_best.pth')
                best_acc = acc
            scheduler.step()
        wandb.save(f'./{args.run_name}/ckpt_best.pth')

sweep_id = wandb.sweep(sweep_config, project=wandb_projname)

wandb.agent(sweep_id, main)