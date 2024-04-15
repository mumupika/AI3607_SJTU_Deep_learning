import jittor as jt
import pygmtools as pygm
from jittor.optim import Optimizer
from jittor import nn
from jittor import Module
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from jittor.dataset.cifar import CIFAR10
from jittor.dataset import DataLoader
import jittor.transform as trans

'''Alexnet definition'''
class AlexNet(Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.c1=nn.Conv(3,96,11,4,1)
        self.bn1=nn.BatchNorm(96)
        self.relu=nn.Relu()
        self.maxpool=nn.MaxPool2d(3,2)
        self.c2=nn.Conv(96,256,5,1,2)
        self.bn2=nn.BatchNorm(256)
        #relu
        #maxpool
        self.c3=nn.Conv(256,384,3,1,1)
        #relu
        self.bn3=nn.BatchNorm(384)
        #self.c4=nn.Conv(384,384,3,1,1)
        #self.bn4=nn.BatchNorm(384)
        #relu
        self.c5=nn.Conv(384,256,3,1,1)
        #relu
        #maxpool
        self.bn5=nn.BatchNorm(256)
        self.flat=nn.Flatten()
        self.l1=nn.Linear(256*5*5,512)
        #relu
        #reshape
        self.l2=nn.Linear(512,64)
        self.l3=nn.Linear(64,4)
    
    def execute(self,x):
        x=self.c1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.c2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.c3(x)
        x=self.bn3(x)
        x=self.relu(x)
        #x=self.c4(x)
        #x=self.bn4(x)
        #x=self.relu(x)
        x=self.c5(x)
        x=self.bn5(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.flat(x)
        x=self.l1(x)
        x=self.relu(x)
        x=self.l2(x)
        x=self.relu(x)
        x=self.l3(x)    #(64,4)
        #sinkhorn,TBD
        x=pygm.linear_solvers.sinkhorn(x)
        return x


def train_image_load(train_data):
    batch_size=0
    imgs=[]# for image to all images set
    for data in train_data:
        img,target=data
        batch_size+=1
        for i in range(len(img)):
            image=[]
            '''Containing all 4 parts.'''
            image.append(img[i].permute(2,1,0)[:,0:224,0:224])
            image.append(img[i].permute(2,1,0)[:,224:224*2,0:224])
            image.append(img[i].permute(2,1,0)[:,0:224,224:224*2])
            image.append(img[i].permute(2,1,0)[:,224:224*2,224:224*2])
            imgs.append(image)
        if batch_size%1==0: #here change batch_size
            imgs=jt.Var(imgs).float64()
            yield imgs
            imgs=[]
    #imgs=jt.Var(imgs).float32()
    #return imgs
    


def target_generation(images):
    '''Randomly shuffle permutation of image,Generate target'''
    rearranged_images=[]
    targets=[]
    for i in range(len(images)):
        permute=np.random.permutation(4)[:4]
        rearranged_img=[]
        target=np.zeros((4,4))
        for j in range(len(images[i])):
            rearranged_img.append(images[i][permute[j]])
            target[j][permute[j]]=1
        rearranged_images.append(rearranged_img)
        targets.append(target)
    
    rearranged_images,targets=jt.Var(rearranged_images).float64(),jt.Var(targets)
    return rearranged_images,targets

def real_train_target(targets_i):
    target=[] #This is the real target
    for i in range(len(targets_i)):
        for j in range(len(targets_i[i])):
            if targets_i[i][j]==1:
                target.append(j)
    target=jt.Var(target).float64()
    return target
        
        
def train(net,optimizer,train_data_loader,epoch):
    train_step=0
    net.train()
    for image in train_image_load(train_data_loader):
        inputs,targets=target_generation(image)
        for i in range(4):
            tar=real_train_target(targets[:,i,:])
            outputs=net(inputs[:,i,:,:,:])
            loss=nn.cross_entropy_loss(outputs,tar)
            optimizer.step(loss)
            train_step+=1
            print(f'epoch:{epoch},Step:{train_step},Loss:{loss}')


def test_image_load(test_data):
    batch_size=0
    imgs=[]
    for data in test_data:
        img,target=data
        batch_size+=1
        for i in range(len(img)):
            image=[]
            '''Containing all 4 parts.'''
            image.append(img[i].permute(2,1,0)[:,0:224,0:224])
            image.append(img[i].permute(2,1,0)[:,224:224*2,0:224])
            image.append(img[i].permute(2,1,0)[:,0:224,224:224*2])
            image.append(img[i].permute(2,1,0)[:,224:224*2,224:224*2])
            imgs.append(image)
            
        imgs=jt.Var(imgs).float64()
        yield imgs
        imgs=[]


def test_target_generation(images):
    '''Randomly shuffle permutation of image,Generate target'''
    rearranged_images=[]
    targets=[]
    for i in range(len(images)):
        permute=np.random.permutation(4)[:4]
        rearranged_img=[]
        target=np.zeros((4,4))
        for j in range(len(images[i])):
            rearranged_img.append(images[i][permute[j]])
            target[j][permute[j]]=1
        rearranged_images.append(rearranged_img)
        targets.append(target)
    
    rearranged_images,targets=jt.Var(rearranged_images).float64(),jt.Var(targets)
    return rearranged_images,targets

def eval(outputs,target_i):
    acc=0
    for i in range(len(outputs)):
        pred=jt.argmax(outputs[i],0)
        real=jt.argmax(target_i[i],0)
        if pred[0]==real[0]:
            acc+=1
    return acc

def test(net,optimizer,test_data_loader,epoch):
    test_step=0
    total_acc=0
    net.eval()
    for image in test_image_load(test_data_loader):
        inputs,targets=test_target_generation(image)
        for i in range(4):
            outputs=net(inputs[:,i,:,:]) # output(64,4),target(64,4,4)
            acc=eval(outputs,targets[:,i,:])
            total_acc+=acc
        total_acc/=4
        test_step+=1
        print(f'epoch:{epoch},Step:{test_step},Accuracy:{total_acc/(len(outputs)*4)*100}%')
            
            
def main():
    '''The target and the full set of images have completed'''
    net=AlexNet()
    learning_rate=1e-5
    optimizer=nn.SGD(net.parameters(),lr=learning_rate)
    net.train()
    #scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.1)
    # 1. Get the train data
    train_data=CIFAR10(train=True,transform=trans.RandomResizedCrop((224*2,224*2)))
    train_data_loader=DataLoader(train_data,batch_size=16)
    test_data=CIFAR10(train=False,transform=trans.RandomResizedCrop((224*2,224*2)))
    test_data_loader=DataLoader(test_data,batch_size=16)
    epochs=100
    for epoch in range(epochs):
        train(net,optimizer,train_data_loader,epoch+1)
        test(net,optimizer,test_data_loader,epoch+1)
    
if __name__== '__main__':
    pygm.set_backend('jittor')
    main()