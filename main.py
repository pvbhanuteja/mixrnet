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
import os
import argparse
from dataloader import  CIFAR_Dataset
from mixup import mixup_cross_entropy_loss
from models import *
from utils import progress_bar, get_time_str
from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--run_name','-rn',type=str, help='your experiment name',default=f'{get_time_str()}_default')
parser.add_argument('--batch_size', '-b', default=512, type=int, help='batch size')
parser.add_argument('--ckp_last', '-cl', default=True, type=bool, help='resume with last checkpoint if false resume with best checkpoint')
parser.add_argument('--num_epochs', '-ne', default=200, type=int, help='number of epochs')

timestamp = get_time_str()
print(timestamp)
args = parser.parse_args()
if args.resume:
    folder_name = args.run_name
else:
    folder_name = f'runs/{args.run_name}/{timestamp}'
writer = SummaryWriter(folder_name)

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
trainset = CIFAR_Dataset('data/cifar-10-batches-py/', True, transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testset = CIFAR_Dataset('data/cifar-10-batches-py/', False, transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.run_name), 'Error: no checkpoint directory found!'
    if args.ckp_last:
        checkpoint = torch.load(f'./{args.run_name}/models/ckpt_last.pth')
    else:
        checkpoint = torch.load(f'./{args.run_name}/models/ckpt_best.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
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
    writer.add_scalar("Loss/train", train_loss/(batch_idx+1), epoch)
    writer.add_scalar("Accuracy/train", 100.*correct/total, epoch)
    for name, weight in net.named_parameters():
        writer.add_histogram(name,weight, epoch)
        writer.add_histogram(f'{name}.grad',weight.grad, epoch)


    if (epoch+1)%10 == 0:
        print('Saving model..')
        state = {
            'net': net.state_dict(),
            'acc': 100.*correct/total,
            'epoch': epoch,
        }
        if not os.path.isdir(f'./{folder_name}/models'):
            os.makedirs(f'./{folder_name}/models')
        torch.save(state, f'./{folder_name}/models/ckpt_last.pth')


def test(epoch):
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
    writer.add_scalar("Loss/test", test_loss/(batch_idx+1), epoch)
    writer.add_scalar("Accuracy/test", acc, epoch)
    if acc > best_acc:
        print('Saving best model..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(f'./{folder_name}/models'):
            os.makedirs(f'./{folder_name}/models')
        torch.save(state, f'./{folder_name}/models/ckpt_best.pth')
        best_acc = acc

if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch+args.num_epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
    writer.add_hparams(
        {"lr": args.lr, "bsize": args.batch_size},
        {
            "accuracy": best_acc
        },
    )
