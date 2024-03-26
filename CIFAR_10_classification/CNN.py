import jittor as jt
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

class CNNnet(Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1=nn.Conv(3,32,5,1,2)
        self.Maxpool=nn.MaxPool2d(2)
        self.conv2=nn.Conv(32,32,5,1,2)
        self.conv3=nn.Conv(32,64,5,1,2)
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(64*4*4,256)
        self.linear2=nn.Linear(256,64)
        self.linear3=nn.Linear(64,10)
        self.sigmoid=nn.Sigmoid()
        self.relu=nn.Relu()
    def execute(self,x):
        x=self.conv1(x)
        x=self.Maxpool(x)
        x=self.conv2(x)
        x=self.Maxpool(x)
        x=self.conv3(x)
        x=self.Maxpool(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.linear2(x)
        x=self.sigmoid(x)
        x=self.linear3(x)
        return x
    
    
def train(net,train_data_loader,optimizer,total_train_step,epoch):
    net.train()
    for data in train_data_loader:
        imgs,targets=data
        imgs,targets=imgs.float32(),targets.float32()
        imgs=imgs.permute(0,3,1,2)
        outputs=net(imgs)
        loss=nn.cross_entropy_loss(outputs,targets)
        optimizer.step(loss)
        total_train_step+=1
        if total_train_step %50 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, total_train_step*32, len(train_data_loader),
                    100. * total_train_step*32 / len(train_data_loader), loss.data[0]))

def test(net,test_data_loader,epoch):
    net.eval()
    total_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(test_data_loader):
        batch_size = inputs.shape[0]
        inputs,targets=inputs.permute(0,3,1,2).float32(),targets.float32()
        outputs = net(inputs)
        pred = np.argmax(outputs.data, axis=1)
        acc = np.sum(targets.data==pred)
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size
        if batch_idx % 10 == 0:
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\tAcc: {:.6f}'.format(epoch, \
                    batch_idx*batch_size, len(test_data_loader),100. * float(batch_idx)*batch_size / len(test_data_loader), acc))
    print ('Total test acc =', total_acc / total_num)
    return total_acc/total_num

def main():
    net=CNNnet()
    train_data=CIFAR10(train=True)
    test_data=CIFAR10(train=False)
    train_data_loader=DataLoader(train_data,batch_size=32)
    test_data_loader=DataLoader(test_data,batch_size=32)
    learning_rate=0.05
    optimizer=nn.SGD(net.parameters(),lr=learning_rate,weight_decay=0.1)
    total_train_step=0
    test_acc=[]
    epochs=10
    for epoch in range(epochs):
        print("epochs:{}".format(epoch+1))
        train(net,train_data_loader,optimizer,total_train_step,epoch+1)
        test_acc.append(test(net, test_data_loader, epoch+1))
    plt.plot(test_acc,'r',label="test_acc")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
if __name__=="__main__":
    main()