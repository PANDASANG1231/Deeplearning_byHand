import time
import numpy as np
import seaborn as sns
import torch
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

import torchvision
from torch.utils import data
from torchvision import transforms


## 定义metrics
class Accumulator():
    """A simple Accumulator which help accumulate a list while evaluation
    """
    def __init__(self, n):
        self.data = [0.] * n
    
    def add(self, *args):
        for arg in args:
            self.data = [a + float(b) for a,b in zip(self.data, arg)]
    
    def reset(self):
        self.data = [0.] * len(self.data)
    
    def __get_item__(self, idx):
        return self.data[idx]




## 定义metric

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def accuracy_iter(model, data_iter):
    """AI is creating summary for accuracy_iter

    Parameters
    ----------
    model : Function of Tensor
        Return a single Tensor
    data_iter : Iterator
        Usually an iterator yields batch data

    Returns
    -------
    List of floats
        return [loss_of_average, accuracy_of_average, false_rate_of_average]
    """
    
    accu = Accumulator(3)
    with torch.no_grad():
        
        for X, y in data_iter:
            y_hat = model(X)
            cnt = len(y)
            acc_t = (y_hat.argmax(axis=1) == y).sum()
            acc_f = (y_hat.argmax(axis=1) != y).sum()
            accu.add([cnt, acc_t, acc_f])
            
    return [x/accu.data[0] for x in accu.data][1:]

def accuracy_iter_gpu(model, data_iter, device=None):
    """AI is creating summary for accuracy_iter

    Parameters
    ----------
    model : Function of Tensor
        Return a single Tensor
    data_iter : Iterator
        Usually an iterator yields batch data

    Returns
    -------
    List of floats
        return [loss_of_average, accuracy_of_average, false_rate_of_average]
    """
    
    if isinstance(model, torch.nn.Module):
      model.eval()
      if not device:
        device = next(iter(model.parameters())).device

    accu = Accumulator(3)
    with torch.no_grad():
        
        for X, y in data_iter:

            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            
            y = y.to(device)
            y_hat = model(X)
            cnt = len(y)
            acc_t = (y_hat.argmax(axis=1) == y).sum()
            acc_f = (y_hat.argmax(axis=1) != y).sum()
            accu.add([cnt, acc_t, acc_f])
            
    return [x/accu.data[0] for x in accu.data][1:]



## Animation Class
from IPython import display

class Animation():
    
    def __init__(self, epoch_show_num, xlim=None, ylim=[0, 10], secondary=True, xlabel="Epoch"):
        
        import matplotlib.pyplot as plt
        self.epoch_show_num = epoch_show_num
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8,4))
        self.ax.set_xticks(range(epoch_show_num))
        self.ax.set_xlabel(xlabel)
        # self.ax.set_ylim(*ylim)
        self.data = {"l": []}
        
        if secondary == True:
            self.ax2 = self.ax.twinx()
            self.data["r"] = []
            
        self.secondary = secondary

    def add_data(self, data, side='l'):
                
        if self.data[side]:
            self.data[side] = [a + [x] for a, x in zip(self.data[side], data)]
        else:
            self.data[side] = [[x] for x in data]
            
        
    def add(self, data_l, data_r=None, legends_l=None, legends_r=None):
                
        if legends_l is None:
            legends_l = list(range(len(data_l)))
        if legends_r is None:
            legends_r = list(range(len(data_r)))
            
        if data_l:
            self.add_data(data_l, "l")
        if data_r:
            self.add_data(data_r, "r")
        
        self.ax.cla()
        self.ax2.cla()
        
        alpha_l = 1
        alpha_r = 1
                
        for data_l_list, label in zip(self.data['l'], legends_l):
            self.ax.plot(range(len(data_l_list)), data_l_list, label=label, alpha=alpha_l, color='b')
            alpha_l /= 2
        
        if self.secondary:
            for data_r_list, label in zip(self.data['r'], legends_r):
                self.ax2.plot(range(len(data_r_list)), data_r_list, label=label, color='r', alpha=alpha_r, linestyle="--")
                self.ax2.set_xticks(range(self.epoch_show_num))
                self.ax2.set_ylim(0.5, 1)
                alpha_r /= 2


        plt.legend()
        
        if self.secondary:
            plt.grid("minor", axis='both')
        else:
            plt.grid("major", axis='both')

        print(legends_l , legends_r)
        print(data_l , data_r)
        
        display.display(self.fig)
        

        display.clear_output(wait=True)



## Trianer

def train_epoch_p1(model, loss, optimizer, train_data_iter, test_data_iter):
    """training function for one epoch, General in MLP style structrue

    Parameters
    ----------
    model : Model
        Use pytoch model or model in pytorch variables
    loss : torch.nn.Module
        Loss function
    optimizer : torch.optims.Optimizer
        Must be torch's Optimizer Class
    train_data_iter : Iterator
        Iterate data in Train
    test_data_iter : Iterator
        Iterate data in Test

    Returns
    -------
    List
        final_metrics = [train_loss, train_accuracy, test_accuracy]
    """
    
    accu = Accumulator(3)
    for batch_X, batch_y in train_data_iter:
        
        batch_y_hat = model(batch_X)
        batch_loss = loss(batch_y_hat, batch_y)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            n = len(batch_y)
            batch_acc = accuracy(batch_y_hat, batch_y)
            accu.add([n, n*batch_loss, batch_acc])

    train_metric = [x / accu.data[0] for x in accu.data][1:]
    test_acc = accuracy_iter(model, test_data_iter)
    final_metrics = train_metric + [test_acc[0]]
    
    return final_metrics



def train_p1(epoch_num, model, loss, optimizer, train_data_iter, test_data_iter):
    """training function, General in MLP style structrue

    Parameters
    ----------
    epoch_num: Int
        Numbers to train
    model : Model
        Use pytoch model or model in pytorch variables
    loss : torch.nn.Module
        Loss function
    optimizer : torch.optims.Optimizer
        Must be torch's Optimizer Class
    train_data_iter : Iterator
        Iterate data in Train
    test_data_iter : Iterator
        Iterate data in Test

    Returns
    -------
    List
        final_metrics = [train_loss, train_accuracy, test_accuracy]
    """
    
    animation = Animation(epoch_show_num=epoch_num, secondary=True)
    
    for _ in range(epoch_num):
        final_metrics = train_epoch_p1(model, loss, optimizer, train_data_iter, test_data_iter)
        animation.add(data_l=[final_metrics[0]], data_r=final_metrics[1:], 
                      legends_l=["train_loss"], legends_r=["train_accuracy", "test_accuracy"])
    
    print(final_metrics)
    
    
def train_epoch_p2(model, loss, optimizer, train_data_iter, test_data_iter, device):
    """training function for one epoch, General in CNN style structrue, will use GPU run model

    Parameters
    ----------
    model : Model
        Use pytoch model or model in pytorch variables
    loss : torch.nn.Module
        Loss function
    optimizer : torch.optims.Optimizer
        Must be torch's Optimizer Class
    train_data_iter : Iterator
        Iterate data in Train
    test_data_iter : Iterator
        Iterate data in Test

    Returns
    -------
    List
        final_metrics = [train_loss, train_accuracy, test_accuracy]
    """

    accu = Accumulator(3)
    time_accu = Accumulator(2)


    for batch_X, batch_y in train_data_iter:
        time_ = time.time()
        batch_X, batch_y = batch_X.to(device=device), batch_y.to(device=device)

        batch_y_hat = model(batch_X)
        batch_loss = loss(batch_y_hat, batch_y)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        time_ = time.time() - time_


        with torch.no_grad():
            n = len(batch_y)
            batch_acc = accuracy(batch_y_hat, batch_y)
            accu.add([n, n*batch_loss, batch_acc])
            time_accu.add([time_, n])




    train_metric = [x / accu.data[0] for x in accu.data][1:]
    test_acc = accuracy_iter_gpu(model, test_data_iter, device)
    final_metrics = train_metric + [test_acc[0]]

    return final_metrics, [x / time_accu.data[0] for x in time_accu.data][1:]

def train_p2(epoch_num, model, loss, lr, train_data_iter, test_data_iter, device, optim_type="SGD"):
    """training function, General in CNN style structrue, will use GPU run model

    Parameters
    ----------
    epoch_num: Int
        Numbers to train
    model : Model
        Use pytoch model or model in pytorch variables
    loss : torch.nn.Module
        Loss function
    lr : Learning rate
    train_data_iter : Iterator
        Iterate data in Train
    test_data_iter : Iterator
        Iterate data in Test

    Returns
    -------
    List
        final_metrics = [train_loss, train_accuracy, test_accuracy]
    """

    def init_weights(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
          torch.nn.init.xavier_uniform_(m.weight)
    
    model.apply(init_weights)
    model.to(device=device)

    if optim_type == "SGD":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    else:
        if lr > 0.01:
            lr = 0.01
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # animation = Animation(epoch_show_num=epoch_num, secondary=True)
    
    for _ in range(epoch_num):
        final_metrics, examplist = train_epoch_p2(model, loss, optimizer, train_data_iter, test_data_iter, device)
        # animation.add(data_l=[final_metrics[0]], data_r=final_metrics[1:], 
        #               legends_l=["train_loss"], legends_r=["train_accuracy", "test_accuracy"])
      
        print(f'loss {final_metrics[0]:.3f}, train acc {final_metrics[1]:.3f}, '
              f'test acc {final_metrics[2]:.3f}')
        
    print(f'Calculation Ability: {examplist[0]:.1f} examples/sec on {str(device)}')
    
    