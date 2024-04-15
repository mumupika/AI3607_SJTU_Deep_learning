import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
import torch.utils.data as Data  # to make Loader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import os
import numpy as np 
import time
import csv
import pygmtools as pygm



class Resnet(nn.Module):
    def __init__(self):
        super(Resnet,self).__init__()
        self.conv1=nn.Conv2d(3,8,3,1,1)
        self.bn1=nn.BatchNorm2d(8)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(8,32,3,1,1)
        self.bn2=nn.BatchNorm2d(32)
        #relu
        self.conv_y1=nn.Conv2d(3,32,3,1,1)
        #relu
        self.flatten=nn.Flatten()
        self.l1=nn.Linear(32*4*4,32)
        #relu
        self.fc=nn.Linear(32,16)
        self.avg=nn.AvgPool2d(16)
        #softmax
    def forward(self,x):
        y=x
        y=self.conv_y1(y)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        # Resplus
        x+=y
        x=self.relu(x)
        x=self.avg(x)
        x=self.flatten(x)
        x=self.l1(x)
        x=self.fc(x)
        x=torch.reshape(x,(-1,4,4))
        x=pygm.linear_solvers.sinkhorn(x)
        #x=nn.softmax(x,dim=1)
        return x


def train_image_load(train_data):
    batch_size=0
    imgs=[]# for image to all images set
    for data in train_data:
        img,target=data
        img=img.numpy()
        batch_size+=1
        for i in range(len(img)):
            image=[]
            '''Containing all 4 parts.'''
            image.append(img[i][:,0:32,0:32])
            image.append(img[i][:,32:32*2,0:32])
            image.append(img[i][:,0:32,32:32*2])
            image.append(img[i][:,32:32*2,32:32*2])
            imgs.append(image)
        if batch_size%4==0: #here change batch_size
            imgs=np.array(imgs)
            imgs= torch.tensor(imgs)
            yield imgs
            imgs=[]
    #imgs=jt.Var(imgs).float32()
    #return imgs
    


def target_generation(images):
    '''Randomly shuffle permutation of image,Generate target'''
    images=images.numpy()
    rearranged_images=[]
    targets=[]
    for i in range(len(images)):
        permute=np.random.permutation(4)[:4]
        rearranged_img=[]
        target=np.zeros((4,4))
        for j in range(len(images[i])):
            rearranged_img.append(images[i][permute[j]])
            target[j][permute[j]]=1
        #rearranged_img=torch.tensor(rearranged_img)
        rearranged_img=np.reshape(rearranged_img,(3,64,64))
        rearranged_images.append(rearranged_img)
        targets.append(target)
    rearranged_images,targets=np.array(rearranged_images),np.array(targets)
    rearranged_images,targets=torch.tensor(rearranged_images),torch.tensor(targets)
    return rearranged_images,targets

        
        
def train(net,optimizer,train_data_loader,epoch,file):
    net.train()
    train_step=0
    total_loss=0
    for image in train_image_load(train_data_loader):
        inputs,targets=target_generation(image)     #(64,3,64,64) vs (64,4,4)
        for i in range(4):
            inputs,targets=inputs.float().to(device),targets.float().to(device)
            outputs=net(inputs)
            optimizer.zero_grad()
            outputs,targets=outputs.float(),targets.float()
            loss=pygm.utils.permutation_loss(outputs,targets)
            loss.backward()
            optimizer.step()
            train_step+=1
            total_loss+=loss
            if train_step%500==0:
                print(f'epoch:{epoch},Step:{train_step},Loss:{loss}')
                format_text=f"epoch:{epoch},Step:{train_step},Loss:{loss}\n"
                file.write(format_text)
    return total_loss/train_step


def test_image_load(test_data):
    batch_size=0
    imgs=[]
    for data in test_data:
        img,target=data
        img=img.numpy()
        batch_size+=1
        for i in range(len(img)):
            image=[]
            '''Containing all 4 parts.'''
            image.append(img[i][:,0:32,0:32])
            image.append(img[i][:,32:32*2,0:32])
            image.append(img[i][:,0:32,32:32*2])
            image.append(img[i][:,32:32*2,32:32*2])
            imgs.append(image)
        imgs=np.array(imgs)
        imgs=torch.tensor(imgs)
        yield imgs
        imgs=[]


def test_target_generation(images):
    '''Randomly shuffle permutation of image,Generate target'''
    images=images.numpy()
    rearranged_images=[]
    targets=[]
    for i in range(len(images)):
        permute=np.random.permutation(4)[:4]
        rearranged_img=[]
        target=np.zeros((4,4))
        for j in range(len(images[i])):
            rearranged_img.append(images[i][permute[j]])
            target[j][permute[j]]=1
        #rearranged_img=torch.tensor(rearranged_img)
        rearranged_img=np.reshape(rearranged_img,(3,64,64))
        rearranged_images.append(rearranged_img)
        targets.append(target)
    rearranged_images,targets=np.array(rearranged_images),np.array(targets)
    rearranged_images,targets=torch.tensor(rearranged_images),torch.tensor(targets)
    return rearranged_images,targets

def eval(outputs,target_i):
    acc=0
    for i in range(len(outputs)):
        pred=torch.argmax(outputs[i],1)
        real=torch.argmax(target_i[i],1)
        for j in range(len(pred)):
            if pred[j]==real[j]:
                acc+=1
    return acc

def test(net,optimizer,test_data_loader,epoch,file):
    test_step=0
    overall_acc=0
    total_acc=0
    net.eval()
    for image in test_image_load(test_data_loader):
        inputs,targets=test_target_generation(image)
        for i in range(4):
            inputs,targets=inputs.float().to(device),targets.float().to(device)
            outputs=net(inputs) # output(64,4),target(64,4,4)
            acc=eval(outputs,targets)
            total_acc+=acc
        total_acc/=4
        test_step+=1
        overall_acc+=total_acc/(len(outputs)*4)
        if test_step%100==0:
            print(f'epoch:{epoch},Step:{test_step},Accuracy:{total_acc/(len(outputs)*4)*100}%')
            format_text=f'epoch:{epoch},Step:{test_step},Accuracy:{total_acc/(len(outputs)*4)*100}%\n'
            file.write(format_text)
    
    print(f'\n epoch:{epoch},Accuracy:{overall_acc/test_step*100}%\n')
    format_text=f'\n epoch:{epoch},Accuracy:{overall_acc/test_step*100}%\n'
    file.write(format_text)
    return overall_acc/test_step*100
    
def main():
    '''The target and the full set of images have completed'''
    net=Resnet().to(device)
    learning_rate=1e-5
    optimizer=optim.SGD(net.parameters(),lr=learning_rate,momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    # 1. Get the train data
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.RandomResizedCrop((32*2,32*2))])
    train_data = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    train_data_loader = Data.DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        # num_workers=2 # ready to be commented(windows)
    )
    test_data = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform,
    )
    test_data_loader = Data.DataLoader(
        test_data,
        batch_size=16,
        shuffle=False,
        # num_workers=2
    )
    epochs=int(100)
    train_loss=[]
    test_acc=[]
    file=open("output_pytorch_easy.txt","a+")
    for epoch in range(epochs):
        train_loss.append(train(net,optimizer,train_data_loader,epoch+1,file))
        test_acc.append(test(net,optimizer,test_data_loader,epoch+1,file))
        scheduler.step()
    
    plt.plot(test_acc,'r',label="test_acc")
    plt.xlabel("Epochs")
    plt.ylabel("acc")
    plt.legend()
    plt.show()
    
    plt.plot(train_loss,'g',label="train_loss")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()    
if __name__== '__main__':
    device = torch.device("mps")
    pygm.set_backend('pytorch')
    main()