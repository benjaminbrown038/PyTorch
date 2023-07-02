import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

'''
MNIST Training Dataset
'''

mnist_train = torchvision.datasets.MNIST(root = '/data', download = True,train = True,transform = transforms.ToTensor())

'''
MNIST Testing Dataset
'''

mnist_test = torchvision.datasets.MNIST(root = '/data', download = True, train = False, transform = transforms.ToTensor())


'''
MNIST Training DataLoader Object (for model)
'''

mnist_train_dataloader = DataLoader(mnist_train,batch_size = 64,shuffle = True)

'''
MNIST Testing DataLoader Object (for model)
'''

mnist_test_dataloader = DataLoader(mnist_test,batch_size = 64, shuffle = True)


'''
MNIST Training Dataset for (pretrained) model
'''

mnist_train1 = torchvision.datasets.MNIST(root = '/data', download = True,train = True,transform = transforms.Compose([transforms.ToTensor(),transforms.Grayscale(3)]))

'''
MNIST Testing Dataset for (pretrained) model
'''

mnist_test1 = torchvision.datasets.MNIST(root = '/data', download = True, train = False, transform = transforms.Compose([transforms.ToTensor(),transforms.Grayscale(3)]))


'''
MNIST Training DataLoader for (pretrained) model
Still needs work on the transform in the dataset object to capture the correct size of images (converting to gray scale) for the pretrained model
'''

mnist_train_dataloader1 = DataLoader(mnist_train,batch_size = 64,shuffle = True)

'''

'''

mnist_test_dataloader1 = DataLoader(mnist_test,batch_size = 64, shuffle = True)

'''

'''

cifar_train = torchvision.datasets.MNIST(root = '/data', download = True, train = True)

'''

'''

cifar_test = torchvision.datasets.MNIST(root = '/data',download = True, train = False)

'''


'''

cifar_train_dataloader = DataLoader(mnist_train, batch_size = 64, shuffle = True)

'''

'''

cifar_test_dataloader = DataLoader(mnist_test, batch_size = 64, shuffle = True)
