"""
该库收集了《动手学深度学习》(PyTorch版)中封装在d2l包内的函数和类，代码使用中文注释。
将该库上传到Kaggle(导入数据集)或Colab(上传Google Drive)后，在开头添加模块搜索路径即可像书中一样正常运行代码。

# Kaggle
import sys
sys.path.append('/kaggle/input/d2l-module')

# Colab
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.append('/content/drive/My Drive')
"""

import os
import re
import sys
import math
import time
import random
import shutil
import hashlib
import tarfile
import zipfile
import requests
import torch
import torchvision
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils import data
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from IPython import display
from collections import defaultdict
from matplotlib_inline import backend_inline

d2l=sys.modules[__name__]

DATA_HUB=dict() # 将数据集名称映射到数据集相关的二元组上,二元组的元素分别为数据集的url和验证文件的sha-1密钥
DATA_URL='http://d2l-data.s3-accelerate.amazonaws.com/'

d2l.DATA_HUB['kaggle_house_train']=(d2l.DATA_URL+'kaggle_house_pred_train.csv','585e9cc93e70b39160e7921475f9bcd7d31219ce')
d2l.DATA_HUB['kaggle_house_test']=(d2l.DATA_URL+'kaggle_house_pred_test.csv','fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
d2l.DATA_HUB['hotdog']=(d2l.DATA_URL+'hotdog.zip','fba480ffa8aa7e0febbb511d181409f899b9baa5')
d2l.DATA_HUB['cifar10_tiny']=(d2l.DATA_URL+'kaggle_cifar10_tiny.zip','2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')
d2l.DATA_HUB['dog_tiny']=(d2l.DATA_URL+'kaggle_dog_tiny.zip','0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')


# Part 01: 图像绘制与实用类

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times=[]
        self.start()

    def start(self):
        """启动计时器"""
        self.tik=time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

class Accumulator:
    """在n个变量上累加"""
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

def use_svg_display():
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5,2.5)):
    """设置matplotlib的图表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize']=figsize

def set_axes(axes,xlabel,ylabel,xlim,ylim,xscale,yscale,legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X,Y=None,xlabel=None,ylabel=None,legend=None,xlim=None,ylim=None,xscale='linear',yscale='linear',fmts=('-','m--','g-.','r:'),figsize=(3.5,2.5),axes=None):
    """绘制数据点"""
    if legend is None:
        legend=[]
    set_figsize(figsize)
    axes=axes if axes else d2l.plt.gca()
    
    def has_one_axis(X):    # 如果X有一个轴,输出True
        return (hasattr(X,'ndim') and X.ndim==1 or isinstance(X,list) and not hasattr(X[0],'__len__'))
    
    if has_one_axis(X):
        X=[X]
    if Y is None:
        X,Y=[[]]*len(X),X
    elif has_one_axis(Y):
        Y=[Y]
    if len(X)!=len(Y):
        X=X*len(Y)
    axes.cla()
    for x,y,fmt in zip(X,Y,fmts):
        if len(x):
            axes.plot(x,y,fmt)
        else:
            axes.plot(y,fmt)
    set_axes(axes,xlabel,ylabel,xlim,ylim,xscale,yscale,legend)

class Animator:
    """在动画中绘制数据"""
    def __init__(self,xlabel=None,ylabel=None,legend=None,xlim=None,ylim=None,xscale='linear',yscale='linear',fmts=('-','m--','g-.','r:'),nrows=1,ncols=1,figsize=(3.5,2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend=[]
        use_svg_display()
        self.fig,self.axes=d2l.plt.subplots(nrows,ncols,figsize=figsize)
        if nrows*ncols==1:
            self.axes=[self.axes,]
        # 使用lambda函数捕获参数
        self.config_axes=lambda: d2l.set_axes(self.axes[0],xlabel,ylabel,xlim,ylim,xscale,yscale,legend)
        self.X,self.Y,self.fmts=None,None,fmts
        
    def add(self,x,y):
        # 向图表中添加多个数据点
        if not hasattr(y,'__len__'):
            y=[y]
        n=len(y)
        if not hasattr(x,'__len__'):
            x=[x]*n
        if not self.X:
            self.X=[[] for _ in range(n)]
        if not self.Y:
            self.Y=[[] for _ in range(n)]
        for i,(a,b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x,y,fmt in zip(self.X,self.Y,self.fmts):
            self.axes[0].plot(x,y,fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


# Part 02: 线性神经网络训练

def synthetic_data(w,b,num_examples):
    """生成y=Xw+b+噪声"""
    X=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(X,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))

def linreg(X,w,b):
    """线性回归模型"""
    return torch.matmul(X,w)+b

def squared_loss(y_hat,y):
    """均方损失"""
    return (y_hat-y.reshape(y_hat.shape))**2/2

def sgd(params,lr,batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

def load_array(data_arrays,batch_size,is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels=['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5): 
    """绘制图像列表"""
    figsize=(num_cols*scale,num_rows*scale) 
    _,axes=d2l.plt.subplots(num_rows,num_cols,figsize=figsize) 
    axes=axes.flatten() 
    for i,(ax,img) in enumerate(zip(axes,imgs)): 
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy()) 
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False) 
        ax.axes.get_yaxis().set_visible(False) 
        if titles:
            ax.set_title(titles[i]) 
    return axes

def get_dataloader_workers(): 
    """使用4个进程来读取数据"""
    return 4

def load_data_fashion_mnist(batch_size,resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
    mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=get_dataloader_workers()))

def accuracy(y_hat,y):
    """计算预测正确的数量"""
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)
    cmp=y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net,torch.nn.Module):
        net.eval()              # 将模型设置为评估模式
    metric=Accumulator(2)       # 正确预测数,预测总数
    for X,y in data_iter:
        metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

def train_epoch_ch3(net,train_iter,loss,updater):
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net,torch.nn.Module):
        net.train()
    # 训练损失总和,训练准确度总和,样本数
    metric=Accumulator(3)
    for X,y in train_iter:
        # 计算梯度并更新参数
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    # 返回训练损失和训练准确度
    return metric[0]/metric[2],metric[1]/metric[2]

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    """训练模型(定义见第3章)"""
    animator=Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0.3,0.9],legend=['train loss','train acc','test acc'])
    for epoch in range(num_epochs):
        train_metrics=train_epoch_ch3(net,train_iter,loss,updater)
        test_acc=evaluate_accuracy(net,test_iter)
        animator.add(epoch+1,train_metrics+(test_acc,))
    train_loss,train_acc=train_metrics
    assert train_loss<0.5,train_loss
    assert train_acc<=1 and train_acc>0.7,train_acc
    assert test_acc<=1 and test_acc>0.7,test_acc

def predict_ch3(net,test_iter,n=6):
    """预测标签(定义见第3章)"""
    for X,y in test_iter:
        break
    trues=d2l.get_fashion_mnist_labels(y)
    preds=d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles=[true+'\n'+pred for true,pred in zip(trues,preds)]
    d2l.show_images(X[0:n].reshape((n,28,28)),1,n,titles=titles[0:n])


# Part 03: 多层感知机

def evaluate_loss(net,data_iter,loss):
    """评估给定数据集上模型的损失"""
    metric=d2l.Accumulator(2)   # 损失的总和,样本数量
    for X,y in data_iter:
        out=net(X)
        y=y.reshape(out.shape)
        l=loss(out,y)
        metric.add(l.sum(),l.numel())
    return metric[0]/metric[1]

def download(name,cache_dir=os.path.join('..','data')):
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB,f"{name}不存在于{DATA_HUB}"
    url,sha1_hash=DATA_HUB[name]
    os.makedirs(cache_dir,exist_ok=True)
    fname=os.path.join(cache_dir,url.split('/')[-1])
    if os.path.exists(fname):
        sha1=hashlib.sha1()
        with open(fname,'rb') as f:
            while True:
                data=f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest()==sha1_hash:
            return fname    # 命中缓存
    print(f"正在从{url}下载{fname}...")
    r=requests.get(url,stream=True,verify=True)
    with open(fname,'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name,folder=None):
    """下载并解压zip/tar文件"""
    fname=download(name)
    base_dir=os.path.dirname(fname)
    data_dir,ext=os.path.splitext(fname)
    if ext=='.zip':
        fp=zipfile.ZipFile(fname,'r')
    elif ext in ('.tar','.gz'):
        fp=tarfile.open(fname,'r')
    else:
        assert False,'只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir,folder) if folder else data_dir

def download_all():
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)


# Part 04: 卷积神经网络训练(GPU)

def try_gpu(i=0):
    """如果存在,则返回gpu(i),否则返回cpu()"""
    if torch.cuda.device_count()>=i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的GPU,如果没有GPU,返回[cpu(),]"""
    devices=[torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

def corr2d(X,K):
    """计算二维互相关运算"""
    h,w=K.shape
    Y=torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=(X[i:i+h,j:j+w]*K).sum()
    return Y

def evaluate_accuracy_gpu(net,data_iter,device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net,nn.Module):
        net.eval()
        if not device:
            device=next(iter(net.parameters())).device
    metric=d2l.Accumulator(2)   # 正确预测的数量,总预测的数量
    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(X,list):
                X=[x.to(device) for x in X] # BERT微调所需的
            else:
                X=X.to(device)
            y=y.to(device)
            metric.add(d2l.accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

def train_ch6(net,train_iter,test_iter,num_epochs,lr,device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m)==nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on',device)
    net.to(device)
    optimizer=torch.optim.SGD(net.parameters(),lr=lr)
    loss=nn.CrossEntropyLoss()
    animator=d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],legend=['train loss','train acc','test acc'])
    timer,num_batches=d2l.Timer(),len(train_iter)
    for epoch in range(num_epochs):
        metric=d2l.Accumulator(3)   # 训练损失之和,训练准确度之和,样本数
        net.train()
        for i,(X,y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X,y=X.to(device),y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l*X.shape[0],d2l.accuracy(y_hat,y),X.shape[0])
            timer.stop()
            train_l=metric[0]/metric[2]
            train_acc=metric[1]/metric[2]
            if (i+1)%(num_batches//5)==0 or i==num_batches-1:
                animator.add(epoch+(i+1)/num_batches,(train_l,train_acc,None))
        test_acc=evaluate_accuracy_gpu(net,test_iter)
        animator.add(epoch+1,(None,None,test_acc))
    print(f'loss {train_l:.3f},train acc {train_acc:.3f},test acc {test_acc:.3f}')
    print(f'{metric[2]*num_epochs/timer.sum():.1f} examples/sec on {str(device)}')

class Residual(nn.Module):
    """残差块"""
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        super().__init__()
        self.conv1=nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3=nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)
    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        Y+=X
        return F.relu(Y)


# Part 05: 计算性能

class Benchmark:
    """用于测量运行时间"""
    def __init__(self,description='Done'):
        self.description=description
    def __enter__(self):
        self.timer=d2l.Timer()
        return self
    def __exit__(self,*args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')

def split_batch(X,y,devices):
    """将X和y拆分到多个设备上"""
    assert X.shape[0]==y.shape[0]
    return (nn.parallel.scatter(X,devices),nn.parallel.scatter(y,devices))

def resnet18(num_classes,in_channels=1):
    """稍加修改的ResNet-18模型"""
    def resnet_block(in_channels,out_channels,num_residuals,first_block=False):
        blk=[]
        for i in range(num_residuals):
            if i==0 and not first_block:
                blk.append(d2l.Residual(in_channels,out_channels,use_1x1conv=True,strides=2))
            else:
                blk.append(d2l.Residual(out_channels,out_channels))
        return nn.Sequential(*blk)
    # 该模型使用了更小的卷积核、步长和填充,并删除了最大汇聚层
    net=nn.Sequential(
        nn.Conv2d(in_channels,64,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    net.add_module("resnet_block1",resnet_block(64,64,2,first_block=True))
    net.add_module("resnet_block2",resnet_block(64,128,2))
    net.add_module("resnet_block3",resnet_block(128,256,2))
    net.add_module("resnet_block4",resnet_block(256,512,2))
    net.add_module("global_avg_pool",nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc",nn.Sequential(nn.Flatten(),nn.Linear(512,num_classes)))
    return net


# Part 06: 计算机视觉

def train_batch_ch13(net,X,y,loss,trainer,devices):
    """用多GPU进行小批量训练"""
    if isinstance(X,list):
        # 微调BERT中所需
        X=[x.to(devices[0]) for x in X]
    else:
        X=X.to(devices[0])
    y=y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred=net(X)
    l=loss(pred,y)
    l.sum().backward()
    trainer.step()
    train_loss_sum=l.sum()
    train_acc_sum=d2l.accuracy(pred,y)
    return train_loss_sum,train_acc_sum

def train_ch13(net,train_iter,test_iter,loss,trainer,num_epochs,devices=d2l.try_all_gpus()):
    """用多GPU进行模型训练"""
    timer,num_batches=d2l.Timer(),len(train_iter)
    animator=d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0,1],legend=['train loss','train acc','test acc'])
    net=nn.DataParallel(net,device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric=d2l.Accumulator(4)   # 4个维度: 储存训练损失,训练准确度,实例数,特点数
        for i,(features,labels) in enumerate(train_iter):
            timer.start()
            l,acc=train_batch_ch13(net,features,labels,loss,trainer,devices)
            metric.add(l,acc,labels.shape[0],labels.numel())
            timer.stop()
            if (i+1)%(num_batches//5)==0 or i==num_batches-1:
                animator.add(epoch+(i+1)/num_batches,(metric[0]/metric[2],metric[1]/metric[3],None))
        test_acc=d2l.evaluate_accuracy_gpu(net,test_iter)
        animator.add(epoch+1,(None,None,test_acc))
    print(f'loss {metric[0]/metric[2]:.3f},train acc {metric[1]/metric[3]:.3f},test acc {test_acc:.3f}')
    print(f'{metric[2]*num_epochs/timer.sum():.1f} examples/sec on {str(devices)}')

def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname,'r') as f:
        # 跳过文件头行(列名)
        lines=f.readlines()[1:]
    tokens=[l.rstrip().split(',') for l in lines]
    return dict(((name,label) for name,label in tokens))

def copyfile(filename,target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir,exist_ok=True)
    shutil.copy(filename,target_dir)

def reorg_train_valid(data_dir,labels,valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    n=collections.Counter(labels.values()).most_common()[-1][1] # 训练数据集中样本最少的类别中的样本数
    n_valid_per_label=max(1,math.floor(n*valid_ratio))  # 验证集中每个类别的样本数
    label_count={}
    for train_file in os.listdir(os.path.join(data_dir,'train')):
        label=labels[train_file.split('.')[0]]
        fname=os.path.join(data_dir,'train',train_file)
        copyfile(fname,os.path.join(data_dir,'train_valid_test','train_valid',label))
        if label not in label_count or label_count[label]<n_valid_per_label:
            copyfile(fname,os.path.join(data_dir,'train_valid_test','valid',label))
            label_count[label]=label_count.get(label,0)+1
        else:
            copyfile(fname,os.path.join(data_dir,'train_valid_test','train',label))
    return n_valid_per_label

def reorg_test(data_dir):
    """在预测期间整理测试集,以方便读取"""
    for test_file in os.listdir(os.path.join(data_dir,'test')):
        copyfile(os.path.join(data_dir,'test',test_file),os.path.join(data_dir,'train_valid_test','test','unknown'))