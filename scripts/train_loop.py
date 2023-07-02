'''
Incorporate Tensorboard into scripts with pytorch code.
'''

import torch 
import torchvision
from torch.utils.tensorboard import SummaryWriter

'''

'''

mnist_train = torchvision.datasets.MNIST(root = '/data', download = True, train = True, transform = transforms.ToTensor())

'''

'''

mnist_train_dataloader = DataLoader(mnist_train,batch_size = 64, shuffle = True)


'''

'''

loss_fn = CrossEntropyLoss()


'''

'''

optimizer = optim.SGD(model.parameters(),lr = .001, momentum = .9)

'''

'''

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

'''

'''

torch.save('model.pkl'))
