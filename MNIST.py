import jittor as jt
from jittor import nn, Module
import numpy as np
import sys, os
import random
import math
from jittor import init
from jittor.dataset.mnist import MNIST
import jittor.transform as trans
import pylab as pl
 
jt.flags.use_cuda = 0 # if jt.flags.use_cuda = 1 will use gpu

class Model (Module):
    def __init__ (self):
        super (Model, self).__init__()
        self.conv1 = nn.Conv (3, 32, 3, 1) # no padding
        self.conv2 = nn.Conv (32, 64, 3, 1)
        self.bn = nn.BatchNorm(64)
 
        self.max_pool = nn.Pool (2, 2)
        self.relu = nn.Relu()
        self.fc1 = nn.Linear (64 * 12 * 12, 256)
        self.fc2 = nn.Linear (256, 10)
    def execute (self, x) :
        x = self.conv1 (x)
        x = self.relu (x)
 
        x = self.conv2 (x)
        x = self.bn (x)
        x = self.relu (x)
 
        x = self.max_pool (x)
        x = jt.reshape (x, [x.shape[0], -1])
        x = self.fc1 (x)
        x = self.relu(x)
        x = self.fc2 (x)
        return x
 
def train(model, train_loader, optimizer, epoch, losses, losses_idx):
    model.train()
    lens = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        outputs = model(inputs)
        loss = nn.cross_entropy_loss(outputs, targets)
        optimizer.step (loss)
        losses.append(loss.data[0])
        losses_idx.append(epoch * lens + batch_idx)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data[0]))
 
def test(model, val_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.data, axis=1)
        acc = np.sum(targets.data==pred)
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size
        print('Test Epoch: {} [{}/{} ({:.0f}%)]\tAcc: {:.6f}'.format(epoch, \
                    batch_idx, len(val_loader),100. * float(batch_idx) / len(val_loader), acc))
    print ('Total test acc =', total_acc / total_num)
 
def main ():
    batch_size = 16
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 5
    losses = []
    losses_idx = []
    train_loader = MNIST(train=True, transform=trans.Resize(28)).set_attrs(batch_size=batch_size, shuffle=True)
    val_loader = MNIST(train=False, transform=trans.Resize(28)) .set_attrs(batch_size=1, shuffle=False)
    model = Model ()
    optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    for epoch in range(epochs):
        train(model, train_loader, optimizer, epoch, losses, losses_idx)
        test(model, val_loader, epoch)
 
    pl.plot(losses_idx, losses)
    pl.xlabel('Iterations')
    pl.ylabel('Train_loss')
    pl.show()
 
    model_path = '/home/root/Python_Demo/JittorMNISTImageClassification/mnist_model.pkl'
    model.save(model_path)
 
if __name__ == '__main__':
    main()
 
