


import matplotlib.pyplot as plt
from fastai.vision.all import *
from torch.nn import functional
import numpy
from fastai.vision.all import *
from fastai.callback.hook import *
import os
import cv2
from torch.nn import functional
from torch.nn import Sequential, Conv2d
from torch.optim import SGD, Adam
from torchvision import models
from torch import optim
import torchvision
import ray
from torchvision import transforms
from torchvision.models import resnet50, alexnet, alexnet, inception_v3
from torch.nn import Sequential
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.optim import SGD


#import dataset
path = untar_data(URLs.PETS)/'images'
# imagedataloaders to import, clean, and label data
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(path,get_image_files(path),
						valid_pct=.2,seed=42,
						label_func = is_cat, item_tfms=Resize(224))
#cnn learner for building a model
learn = cnn_learner(dls,resnet34, metrics = error_rate)
learn.fine_tune(1)


path = untar_data(URLs.MNIST_SAMPLE)
time = torch.arange(0,20)
params = torch.randn(3).requires_grad_()


def apply_step(params,prn=True):
    speed = time*3 + (time-9.5)**2 + 1
    a,b,c = params
    pred = a*(time**2) + b*time + c
    loss = ((pred - speed)**2).mean()
    loss.backward()
    lr = 1e-5
    # becomes a tensor that computes
    params.grad
    params.data -= lr * params.grad.data
    params.grad=None
    if prn: print(loss.item())
    return pred

# loss
def L1_loss(average,real):
    result = (average - real).abs().mean()
    return result

# loss
def mean_sq_error_loss(average,real):
    result = ((average-real)**2).sqrt().mean()
    return result

# weigths
def init_params(size,std=1.0):
    params = (torch.randn(size)*std).requires_grad_()
    return params

# train
def linear1(xb):
    weights = xb@weights + bias
    return weights

# activation
def sigmoid(x):
    sig = 1/(1+torch.exp(-x))
    return sig

# loss
def mnist_loss(predictions,targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1,1-predictions,predictions).mean()

# train
def calc_grad(xb,yb,model):
    preds = model(xb)
    loss = mnist_loss(preds,yb)
    loss.backward()

# metrics
def batch_accuracy(xb,yb):
    preds = xb.sigmoid()
    correct = (preds > .5) == yb
    result = correct.float().mean()
    return result

# metrics
def validate_epoch(model):
    accs = [batch_accuracy(model(xb),yb) for xb,yb in valid_dl]
    result = round(torch.stack(accs).mean().item(),4)
    return result

# train
def train_epoch(model,dl,opt):
    for xb,yb in dl:
        calc_grad(xb,yb,model)
        opt.step()
        opt.zero_grad()

# train
def train_model(model,epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model),end=' ')

# train
def simple_net(xb):
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res

# train
class BasicOptim:

    def __init__(self,params,lr):
        self.params,self.lr = list(params),lr

    def step(self,*args,**kwargs):
        for p in self.params:
            p.data-=p.grad.data *self.lr

    def zero_grad(self,*args,**kwargs):
        for p in self.params:


# loading data
def load_data(folder_name):
    training_tensor = [tensor(Image.open(i)) for i in folder_name]
    training_stack = ((torch.stack(training_tensor)).float())
    return training_stack

# transforming data
def training_data(*args):
    training = (torch.cat(args))
    return training

# data information
def size(training_stack):
    size = ((training_stack.shape)[1]) * (training_stack.shape[2])
    return size

# creating data
def init_weights(size):
    weights = (torch.randn(size)).requires_grad_()
    return weights

# creating data
def bias():
    bias = torch.randn(1)
    return bias

# transforming data
def transform_data_for_model(training_stack):
    result = training_stack[1] * training_stack[2]
    return result

# transforming data
def matrix_multiply(training_stack):
    new_training_stack = (training_stack).view(-1,784)
    pred = ((new_training_stack) @ weights) + bias
    return pred

# metric
def loss(pred,target):
    loss = (pred-target).abs().mean()
    return loss

# train
def update(lr):
    new_weights -= weights.grad * lr
    return new_weights

# data information
def size_of_image(image):
    image_size = image.shape
    return image_size

# data transformation
def apply_kernel(row,col,kernel):
    convolution = (img[row-1:row+2,col-1:col+2] * kernel).sum()
    return convolution

# transformation
def convolution_top():
    rng = (1,27)
    top_edge = tensor([[apply_kernel(i,j,top_edge) for j in rng] for i in rng])
    return top_edge

# information
def row(padding, stride, height):
    new_row = (height + padding) // stride
    return new_row

# information
def column(padding,stride,height):
    new_column = (height + padding) // stride
    return new_column

# information
def output_shape(w,n,p,f):
    output = int((W - K + (2*P))/(S + 1))
    new_output = (w - n + (2*p) - f) + 1
    return new_output

# creating kernels
def top_edge():
    top_edge = (tensor([1,1,1],[0,0,0],[-1,-1,-1])).float()
    return top_edge

# creating kernels
def bottom_edge():
    bottom_edge = (tensor([-1,-1,-1],[0,0,0],[1,1,1])).float()
    return bottom_edge

# creating kernels
def right_edge():
    right_edge = (tensor([-1,0,1],[-1,0,1],[-1,0,1])).float()
    return right_edge

# creating kernels
def left_edge():
    left_edge = (tensor([1,0,-1],[1,0,-1],[1,0,-1])).float()
    return left_edge

# creating kernels
def diag1_edge():
    diag1_edge = (tensor([1,0,-1],[0,1,0],[-1,0,1])).float()
    return diag1_edge

mnist_train = torchvision.datasets.MNIST(root = '/data', download = True,train = True,transform = transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST(root = '/data', download = True, train = False, transform = transforms.ToTensor())

mnist_train_dataloader = DataLoader(mnist_train,batch_size = 64,shuffle = True)
mnist_test_dataloader = DataLoader(mnist_test,batch_size = 64, shuffle = True)


cifar_train = torchvision.datasets.MNIST(root = '/data', download = True, train = True, batch_size = True, shuffle = True)
cifar_test = torchvision.datasets.MNIST(root = '/data',download = True, train = False, batch_size = True, shuffle = True)


cifar_train_dataloader = DataLoader(mnist_train, batch_size = 64, shuffle = True)
cifar_test_dataloader = DataLoade(mnist_test, batch_size = 64, shuffle = True)


print("Training set of mnist: ", "\n")
print("Type of object that holds training data: ")
print(type(mnist_train),"\n")
print(type(mnist_train[0]))
print("Length of Tuple in torchvision object: ","\n",len(mnist_train[0]),"\n")
print("Image: ")
print(type(mnist_train[0][0]), "\n")
print("Shape of Image: ", "\n", mnist_train[0][0].shape,"\n")
print("Class: ")
print(type(mnist_train[0][1]))
print("Length of training set: ", len(mnist_train[0][0], "\n"))
mnist_train


print("Testing set of MNIST: ", "\n")
print("Type of object that holds testing of data: ")
print(type(mnist_train,"\n"))
print(type(mnist_test,"\n"))
print(type(mnist_test[0]))
print("Length of Tuple in torchvision object ", "/", len(mnist_test[0]),"\n")
print("Image: ")
print(type(mnist_test[0][0]),"\n")
print("shape of Image", "\n", mnist_test[0][0].shape,"\n")
print("Class: ")
print(type(mnist_test[0][1]))
print("Length of training set: ", len(mnist_test[0][0]), "\n")
mnist_test


print("Training set of cifar: ", "\n")
print("Type of object that holds training data: ")
print(type(cifar_train),"\n")
print(type(cifar_train[0]))
print("Length of Tuple in torchvision object ", "\n", len(cifar_train[0],"\n"))
print("Image: ")
print(type(cifar_train[0][0]),"\n")
print("Shape of Image: ", "\n", cifar_train[0][0].shape,"\n")
print("Class: ")
print(type(cifar_train[0][1]))
print("Length of training set: ", len(cifar_train[0][0]), "\n")
cifar_train

print("Testing set of cifar: ", "\n")
print("Type of object that holds testing data: ")
print(type(cifar_test), "\n")
print(type(cifar_test), "\n")
print(type(cifar_test[0]), "\n")
print("Length of Tuple in torchvision object: ", "\n", len(cifar_test[0]), "\n")
print("Image: ")
print(type(cifar_test[0][0]),"\n")
print("Shape of Image:", "\n", cifar_test[0][0].shape, "\n")
print("Class: ")
print(type(cifar_test[0][1]))
print("Length of training set: ", len(cifar_test[0][0]), "\n")
cifar_test

model = nn.Sequential(
			nn.Conv2d(1,32,kernel_size = 3),
			nn.Conv2d(32,64,kernel_szie = 3),
			nn.MaxPool2d(2),
			nn.Dropout(0.25),
			nn.Flatten(),
			nn.Linear(9216,128)
			nn.Linear(128,10),
			nn.Softmax()
			)
summary(model,1,28,28)


model1 = nn.Sequential(
		nn.Conv2d(1,32,kernel_size = 2),
		nn.MaxPool2d(2,2),
		nn.Conv2d(32,64,kernel_size = 2),
		nn.MaxPool2d(2,2),
		nn.Conv2d(64,128,kernel_size = 2),
		nn.Dropout(0.25),
		nn.MaxPool2d(1,1),
		nn.Conv2d(128,10,kernel_size = 3),
		nn.Flatten(),
		nn.Linear(90,10)
		)
summary(model,(1,28,28))

mdoel2 = nn.Sequential(
		nn.Conv2d(1,32,kernel_size = 3),
		nn.Conv2d(32,64,kernel_size = 3),
		nn.MaxPool2d(2),
		nn.Dropot(0.25),
		nn.Flatten(),
		nn.Linear(9216,128),
		nn.Linear(128,10)
		)
summary(model2,(1,28,28))


pre_trained_one = resnet50(pretrained = True)
pre_trained_two = inception_v3(pretrained=True)
pre_trained_three = alexnet(pretrained = True)


size = len(dataloader.dataset)
for batch, (x,y) in enumerate(dataloader):
	pred = model(x)
	loss = loss_fn(pred,y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if batch % 100 == 0:
		loss, current = loss.item(), batch * len(x)


def train_loop(dataloader,model,loss_fn,optimizer):
	size = len(dataloader.dataset)
	for batch, (x,y) in enumerate(dataloader):
		pred = model(x)
		loss = loss_fn(pred,y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if batch % 100 == 0:
			loss,current = loss.item(), batch*len(x)

loss = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(),lr = 0.01,momentum = 0.9)
optimizer1 = optim.SGD(model1.parameters(),lr=0.01,momentum = 0.9)
optimizer2 = optim.SGD(model2.parameters(), lr =0.01,momentum = 0.9)
optimizer3 = optim.SGD(pre_trained_one.parameters(),lr=0.01,momentum = 0.9)
optimizer4 = optim.SGD(pre_trained_two.parameters(),lr=0.01,momentum=0.9)
optimizer5 = optim.SGD(pre_trained_three.parameters(),lr=0.01,momentum=0.9)

train_loop(mnist_train_dataloader,model,loss,optimizer)
train_loop(mnist_train_dataloader,model1,loss,optimizer1)
train_loop(mnist_train_dataloader,model2,loss,optimizer2)
train_loop(mnist_train_dataloader,pre_trained_one,loss,optimizer3)
train_loop(mnist_train_dataloader,pre_trained_two,loss,optimizer4)
train_loop(mnist_train_dataloader,pre_trained_three,loss,optimizer5)
