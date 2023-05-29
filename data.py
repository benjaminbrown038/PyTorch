from torch.utils.data import DataLoader

mnist_train = torchvision.datasets.MNIST(root = '/data', download = True,train = True,transform = transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST(root = '/data', download = True, train = False, transform = transforms.ToTensor())

mnist_train_dataloader = DataLoader(mnist_train,batch_size = 64,shuffle = True)
mnist_test_dataloader = DataLoader(mnist_test,batch_size = 64, shuffle = True)


mnist_train1 = torchvision.datasets.MNIST(root = '/data', download = True,train = True,transform = transforms.Compose([transforms.ToTensor(),transforms.Grayscale(3)]))
mnist_test1 = torchvision.datasets.MNIST(root = '/data', download = True, train = False, transform = transforms.Compose([transforms.ToTensor(),transforms.Grayscale(3)]))

mnist_train_dataloader1 = DataLoader(mnist_train,batch_size = 64,shuffle = True)
mnist_test_dataloader1 = DataLoader(mnist_test,batch_size = 64, shuffle = True)


cifar_train = torchvision.datasets.MNIST(root = '/data', download = True, train = True)
cifar_test = torchvision.datasets.MNIST(root = '/data',download = True, train = False)


cifar_train_dataloader = DataLoader(mnist_train, batch_size = 64, shuffle = True)
cifar_test_dataloader = DataLoader(mnist_test, batch_size = 64, shuffle = True)
