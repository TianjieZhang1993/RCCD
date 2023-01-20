from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision
import argparse
import os
import torch
import torchvision.models as models
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import glob
import random
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR', default='imagenet',
#                     help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument("--input_size",type=int,default=256)
parser.add_argument("--root_path",type=str,default='C:\\Users\\tjzhang\\Documents\\GitHub\\RCCdataset\\crackdata')
best_acc1 = 0

args, unknown = parser.parse_known_args()

print(args)

#preprocess. Run once to split the test, train and validation
#-----------------------------------------------------------------------#
# dirs=glob.glob(os.path.join(args.root_path,'*'))
# dirs = [d for d in dirs if os.path.isdir(d)]#only path
#
#
# train_ratio=0.6
# val_ratio=0.2
# test_ratio=0.2
#
#
# for path in dirs:
#     print(path)
#     path =path.split("\\")[-1]
#     print(path)
#     os.makedirs(f'train\\{path}',exist_ok=True)
#     os.makedirs(f'val\\{path}',exist_ok=True)
#     os.makedirs(f'test\\{path}',exist_ok=True)
#     files=glob.glob(os.path.join(args.root_path,path,'*.png'))
#
#     files+=glob.glob(os.path.join(args.root_path,path,'*.jpg'))
#
#     random.shuffle(files)
#
#     #print(len(files))
#
#     trainfiles=int(len(files)*train_ratio)
#     valfiles=int(len(files)*val_ratio)
#     testfiles=int(len(files)*test_ratio)
#
#     for i, file in enumerate(files):
#         #print(i)
#         #print(file)
#         img=Image.open(file)
#         if i <=trainfiles:
#             img.save(os.path.join(f'train\\{path}',file.split('\\')[-1].split('.')[0]+'.jpg'))
#         elif i <=trainfiles+valfiles:
#             img.save(os.path.join(f'val\\{path}',file.split('\\')[-1].split('.')[0]+'.jpg'))
#         else:
#             img.save(os.path.join(f'test\\{path}',file.split('\\')[-1].split('.')[0]+'.jpg'))
#
# train_files=glob.glob(os.path.join('train','*','*.jpg'))
# val_files=glob.glob(os.path.join('val','*','*.jpg'))
# test_files=glob.glob(os.path.join('test','*','*.jpg'))
#
# print(f'Totally {len(train_files)} files for training')
# print(f'Totally {len(val_files)} files for validating')
# print(f'Totally {len(test_files)} files for testing')
#-------------------------------------------------------------------------------------------------#

# Loading and normalizing the data.
# Define transformations for the training and test sets
# https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model


# Create an instance for training.

def build_transform(is_train, args):
    if is_train:
        print('train transform')
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((args.input_size, args.input_size)),
                # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(0.5),
                # torchvision.transforms.RandomPerspective(distortion_scale=0.6,p=1.0),
                # torchvision.transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

            ]

        )
    print('eval transform')
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.input_size, args.input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),

        ]
    )


def build_dataset(is_train, args, dirr):
    transform = build_transform(is_train, args)
    path = os.path.join(args.root_path, dirr)
    print(path)

    # path=os.path.join(args.root_path,'train' if is_train else 'test')
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    info = dataset.find_classes(path)
    print(f'finding classes from {path}:\t{info[0]}')  # can find what classes in the folder
    print(f'mapping classes from {path} to indexes:\t{info[1]}')
    return dataset, info


train_set, classes = build_dataset(True, args, 'train')

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
print("The number of images in a training set is: ", len(train_loader) * args.batch_size)

val_set, classes = build_dataset(False, args, 'val')

val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
print("The number of images in a validating set is: ", len(val_loader) * args.batch_size)

test_set, classes = build_dataset(False, args, 'test')

test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

print("The number of images in a test set is: ", len(test_loader) * args.batch_size)

print("The number of batches per epoch is: ", len(train_loader))

print(train_set)
#-------------------------------------------# model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
net=models.alexnet(pretrained=True)

#net=models.alexnet(pretrained=True)
#net=models.vgg16(pretrained=True)
print(net)
for param in net.parameters():
  param.requires_grad=False
  net.classifier._modules['6'] = nn.Linear(4096, 4)#for vgg16, alexnet
  #net.fc=nn.Linear(512,4)#for resnet18
net.to(device)



criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)
"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)



start = time.time()
running_losses = []
acc_val = []
acc_test = []
for epoch in range(300):  # loop over the dataset multiple times
    print('=====The %d epoch====' % epoch)
    for i, data in enumerate(train_loader, 0):
        running_loss = 0.0
        # get the inputs; data is a list of [inputs, labels]
        #######inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print(loss.item())

        running_loss += loss.item()

    running_losses.append(running_loss)

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = correct / total

        acc_val.append(accuracy)

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = correct / total

        acc_test.append(accuracy)

end = time.time()
print('Finished Training')
running_time = end - start

print('running time is :', running_time)

import pandas as pd
data_collect=pd.DataFrame({'running_losses':running_losses,'acc_val':acc_val,'acc_test':acc_test})
data_collect.to_csv('alexnet-1.csv')
print('success')
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

correct = 0
total = 0
test_preds = []
test_trues = []
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # print(torch.max(outputs.data, 1))
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        # print(predicted)
        test_preds.extend(predicted.detach().cpu().numpy())
        # print(test_preds)
        test_trues.extend(labels.detach().cpu().numpy())
        # print(test_trues)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    sklearn_precision = precision_score(test_trues, test_preds, average='micro')
    sklearn_recall = recall_score(test_trues, test_preds, average='micro')
    sklearn_f1 = f1_score(test_trues, test_preds, average='micro')
    print(classification_report(test_trues, test_preds))
    print(confusion_matrix(test_trues, test_preds))
    # print(conf_matrix)
    # plot_confusion_matrix(conf_matrix)
    # print("[sklearn_metrics] accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1))
    print(f'Accuracy of the network on test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes[0]}
total_pred = {classname: 0 for classname in classes[0]}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[0][label]] += 1
            total_pred[classes[0][label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


