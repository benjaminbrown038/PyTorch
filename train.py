import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, Softmax, CrossEntropyLoss


model = nn.Sequential(
			nn.Conv2d(1,32,kernel_size = 3),
			nn.Conv2d(32,64,kernel_szie = 3),
			nn.MaxPool2d(2),
			nn.Dropout(0.25),
			nn.Flatten(),
			nn.Linear(9216,128),
			nn.Linear(128,10),
			nn.Softmax())

mnist_train = torchvision.datasets.MNIST(root = '/data', download = True, train = True, transform = transforms.ToTensor())
mnist_train_dataloader = DataLoader(mnist_train,batch_size = 64, shuffle = True)
loss_fn = CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = .001, momentum = .9)

def train_loop(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    for batch, (x,y) in enumerate(dataloader):
        pred = model(x)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss,current = loss.item(),batch*len(x)

PATH = '/path'
torch.save('model.pkl',PATH)
