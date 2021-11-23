from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
from dataloader import  CIFAR_Dataset
from private_loader import Private_Dataset
from mixup import mixup_cross_entropy_loss
from models import *
from utils import progress_bar, get_time_str

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

net = ResNet50PreAct()
net = net.to(device)
net = torch.nn.DataParallel(net)
checkpoint = torch.load(f'./saved_models/9457.ckpt.pth',map_location=torch.device(device))
net.load_state_dict(checkpoint['net'])
privateset = Private_Dataset('./data/cifar-10-batches-py/private_test_images_v3.npy', transform_test)
privateloader = torch.utils.data.DataLoader(privateset, batch_size=1, shuffle=False, num_workers=2)
if __name__=='__main__':
    net.eval()
    with torch.no_grad():
        preds = []
        preds_prob = []
        for batch_idx, (inputs,label) in enumerate(privateloader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            predicted = outputs.data.max(1, keepdim=True)[1]
            preds.append(predicted[0].cpu().detach().numpy()[0])
            preds_prob.append(outputs.cpu().detach().numpy()[0])
        preds = np.array(preds)
        print(preds)
        preds_prob = np.array(preds_prob)
        e = np.exp(preds_prob- np.max(preds_prob))
        S = np.sum(e,axis=1)
        P = e/np.expand_dims(S, 1)
        np.save('preds_final.npy',preds)
        np.save('pred_probs_final.npy',P)