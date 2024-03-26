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
        self.model=nn.Sequential(
            nn.Conv(3,8,3,1,1),
            nn.BatchNorm(8),
            nn.Relu(),
            nn.MaxPool2d(2),
            nn.Conv(8,16,3,1,1),
            nn.BatchNorm(16),
            nn.Relu(),
            nn.MaxPool2d(2),
            nn.Conv(16,32,3,1,1),
            nn.BatchNorm(32),
            nn.Relu(),
            nn.Conv(32,64,3,1,1),
            nn.BatchNorm(64),
            nn.Relu(),
            nn.MaxPool2d(2),
            nn.Conv(64,128,3,1,1),
            nn.BatchNorm(128),
            nn.Relu(),
            nn.Flatten(),
            nn.Linear(128*4*4,512),
            nn.BatchNorm(512),
            nn.Relu(),
            nn.Linear(512,10),
            nn.Softmax(dim=1)
        )
        
    def execute(self,x):
        x=self.model(x)
        return x
    
    
def train(net,train_data_loader,optimizer,total_train_step,epoch,compose):
    net.train()
    for data in train_data_loader:
        imgs,targets=data
        imgs,targets=imgs.float32(),targets.float32()
        imgs=imgs.permute(0,3,1,2)
        imgs=compose(imgs)
        outputs=net(imgs)
        loss=nn.cross_entropy_loss(outputs,targets)
        optimizer.step(loss)
        total_train_step+=1
        if total_train_step %50 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, total_train_step*32, len(train_data_loader),
                    100. * total_train_step*32 / len(train_data_loader), loss.data[0]))

def test(net,test_data_loader,epoch,compose):
    net.eval()
    total_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(test_data_loader):
        batch_size = inputs.shape[0]
        inputs,targets=inputs.permute(0,3,1,2).float32(),targets.float32()
        inputs=compose(inputs)
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
    compose=trans.Compose([trans.ImageNormalize((0.485,0.456,0.406),(0.229, 0.224, 0.225))])
    train_data_loader=DataLoader(train_data,batch_size=32)
    test_data_loader=DataLoader(test_data,batch_size=32)
    learning_rate=0.1
    optimizer=nn.SGD(net.parameters(),lr=learning_rate,momentum=0.9,weight_decay=5e-4)
    scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    total_train_step=0
    total_test_step=0
    test_acc=[]
    epochs=100
    for epoch in range(epochs):
        print("epochs:{}".format(epoch+1))
        train(net,train_data_loader,optimizer,total_train_step,epoch+1,compose)
        test_acc.append(test(net, test_data_loader, epoch+1,compose))
        scheduler.step()
    
    plt.plot(test_acc,'r',label="test_acc")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()    
if __name__=="__main__":
    main()