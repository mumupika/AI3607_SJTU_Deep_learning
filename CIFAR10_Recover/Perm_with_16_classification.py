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

'''Resnet definition'''
'''Todo'''
class Resnet(Module):
    def __init__(self):
        super(Resnet,self).__init__()
        self.conv1=nn.Conv(3,8,3,1,1)
        self.bn1=nn.BatchNorm(8)
        self.relu=nn.Relu()
        self.maxpool=nn.MaxPool2d(2)
        self.conv2=nn.Conv(8,16,3,1,1)
        self.bn2=nn.BatchNorm(16)
        #relu
        #maxpool
        self.conv3=nn.Conv(16,32,3,1,1)
        self.bn3=nn.BatchNorm(32)
        #relu
        #maxpool
        self.conv4=nn.Conv(32,64,3,1,1)
        self.bn4=nn.BatchNorm(64)
        #relu
        #maxpool
        self.conv5=nn.Conv(64,128,3,1,1)
        self.bn5=nn.BatchNorm(128)
        #relu
        self.flatten=nn.Flatten()
        self.l1=nn.Linear(128*4*4,512)
        self.bn6=nn.BatchNorm(512)
        #relu
        self.fc=nn.Linear(512,16)
        #softmax
    def execute(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.conv4(x)
        x=self.bn4(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.conv5(x)
        x=self.bn5(x)
        x=self.relu(x)
        x=self.flatten(x)
        x=self.l1(x)
        x=self.bn6(x)
        x=self.relu(x)
        x=self.fc(x)
        x=jt.reshape(x,(-1,4,4))
        x=pygm.linear_solvers.sinkhorn(x)
        #x=nn.softmax(x,dim=1)
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
            image.append(img[i].permute(2,1,0)[:,0:32,0:32])
            image.append(img[i].permute(2,1,0)[:,32:32*2,0:32])
            image.append(img[i].permute(2,1,0)[:,0:32,32:32*2])
            image.append(img[i].permute(2,1,0)[:,32:32*2,32:32*2])
            imgs.append(image)
        if batch_size%4==0: #here change batch_size
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
        rearranged_img=jt.Var(rearranged_img).permute(1,0,2,3)
        rearranged_img=jt.reshape(rearranged_img,(3,64,64))
        rearranged_images.append(rearranged_img)
        targets.append(target)
    
    rearranged_images,targets=jt.Var(rearranged_images).float64(),jt.Var(targets)
    return rearranged_images,targets

        
        
def train(net,optimizer,train_data_loader,epoch,file):
    net.train()
    train_step=0
    total_loss=0
    for image in train_image_load(train_data_loader):
        inputs,targets=target_generation(image)     #(64,3,64,64) vs (64,4,4)
        for i in range(4):
            outputs=net(inputs)
            loss=pygm.utils.permutation_loss(outputs,targets)
            optimizer.step(loss)
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
        batch_size+=1
        for i in range(len(img)):
            image=[]
            '''Containing all 4 parts.'''
            image.append(img[i].permute(2,1,0)[:,0:32,0:32])
            image.append(img[i].permute(2,1,0)[:,32:32*2,0:32])
            image.append(img[i].permute(2,1,0)[:,0:32,32:32*2])
            image.append(img[i].permute(2,1,0)[:,32:32*2,32:32*2])
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
        rearranged_img=jt.Var(rearranged_img).permute(1,0,2,3)
        rearranged_img=jt.reshape(rearranged_img,(3,64,64))
        rearranged_images.append(rearranged_img)
        targets.append(target)
    
    rearranged_images,targets=jt.Var(rearranged_images).float64(),jt.Var(targets)
    return rearranged_images,targets

def eval(outputs,target_i):
    acc=0
    for i in range(len(outputs)):
        pred=jt.argmax(outputs[i],1)
        real=jt.argmax(target_i[i],1)
        for j in range(len(pred[0])):
            if pred[0][j]==real[0][j]:
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
    net=Resnet()
    learning_rate=1e-5
    optimizer=nn.SGD(net.parameters(),lr=learning_rate,momentum=0.9)
    scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    # 1. Get the train data
    train_data=CIFAR10(train=True,transform=trans.RandomResizedCrop((32*2,32*2)))
    train_data_loader=DataLoader(train_data,batch_size=16)
    test_data=CIFAR10(train=False,transform=trans.RandomResizedCrop((32*2,32*2)))
    test_data_loader=DataLoader(test_data,batch_size=16)
    epochs=int(1e5)
    train_loss=[]
    test_acc=[]
    file=open("output.txt","a+")
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
    pygm.set_backend('jittor')
    main()