


import matplotlib.pyplot as plt
from torch.nn import functional
import numpy
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
