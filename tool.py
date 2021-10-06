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
def accuracy_iter(model, data_iter, funt_loss):
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
    
    accu = Accumulator(4)
    with torch.no_grad():
        
        for X, y in data_iter:
            y_hat = model(X)
            cnt = len(y)
            loss = funt_loss(y_hat, y)
            acc_t = (y_hat.argmax(axis=1) == y).sum()
            acc_f = (y_hat.argmax(axis=1) != y).sum()
            accu.add([cnt, loss, acc_t, acc_f])
            
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

        
        display.display(self.fig)
        
        display.clear_output(wait=True)
