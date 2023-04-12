from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.otpim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

def load_data(data_dir = "./data"):
    transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                ])
    trainset = torchvision.datasets.CIFAR(root = data_dir, train = True,download = True,transform = transform)
    testset = torchvision.datasets.CIFAR10(root = data_dir,train = True, download = True, transform = transform)

    return trainset, testset

class Net(nn.Module):
    def __init__(self,l1 = 120,l2 = 84):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,l1)
        self.fc2 = nn.Linear(l1,l2)
        self.fc3 = nn.Linear(l2,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net(config["l1"],config["l2"])

if checkppoint_dir:
    model_state, optimizer_state = torch.load(os.path.join(checkppoint_dir,"checkpoint"))
    net.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

optimizer = optim.SGD(net.parameters(),lr = config["lr"],momentum = 0.9)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        net = nn.DataParellel(net)
net.to(device)



for i, data in enumerate(trainloader,0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)

with tune.checkpoint_dir(epoch) as checkpoint_dir:
    path = os.path.join(checkpoint_dir,"checkpoint")
    torch.save((net.state_dict(), optimizer.state_dict()),path)

tune.report(loss = (val_loss / val_steps,accuracy = correct / total))

def train_cifar(config, checkpoint_dir = None,data_dir = None):
    net = Net(config["l1"],config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParellel(net)
        net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = config["lr"], momentum = .9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir,"checkpoint"))
        net.load_state_dict(model_state)
        optimzer = optim.SGD(net.parameters(),lr = config["lr"],momentum = .9)

    trainset, testset = load_data(data_dir)
    train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size = int(config["batch_size"]),
        shuffle = True,
        num_workers = 8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size = int(config["batch_size"]),
        shuffle = True,
        num_workers = 8)
    for epoch in range(10):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader,0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimzer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    for i, data in enumerate(valloader,0):
        with torch.no_grad():
            inputs,labels = data
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            val_loss += loss.cpu().numpy()
            val_steps += 1

    with tune.checkpoint_dir(epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir,"checkpoint")
        torch.save((net.state_dict(),optimzer.state_dict()),path)

    tune.report(loss= (val_loss/val_steps),accuracy = correct / total)

    print("Finished")

def test_accuracy(net,device = "cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(testset,batch_size = 4, shuffle = False, num_workers = 2)

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images,labels = data
            images, labels - images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(ouputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

    config = {
        "l1": tune.sample_from(lambda _: 2**np.random.randint(2,9)),
        "l2": tune.sample_from(lambda _:2**np.rnadom.randint(2,9)),
        "lr": tune.loguniform(1e-4,1e-1),
        "batch_size": tune.choice([2,4,8,16])
    }

    gpus_per_trial = 2
    result = tune.run(partial(train_cifar,data_dir=data_dir),
                        resources_per_trial = {"cpu": 8 , "gpu":gpus_per_trial},
                        config=config,
                        num_samples = num_samples,
                        scheduler = scheduler,
                        progress_reporter = reporter,
                        checkpoint_at_end = True)


    def main(num_samples = 10, max_num_epochs=10, gpus_per_trial = 2):
        data_dir = os.path.abspath("./data")
        load_data(data_dir)
        config = {
            "l1": tune.sample_from(lambda _: 2** np.random.randint(2,9)),
            "l2": tune.sample_from(lambda _: 2** np.random.randint(2,9)),
            "lr": tune.loguniform(1e-4,1e-1),
            "batch_size": tune.choice([2,4,8,16])
        }
        scheduler = ASHAScheduler(
                        metric = "loss",
                        mode = "min",
                        max_t = max_num_epochs,
                        grace_period = 1,
                        reduction_factor = 2)

        reporter = CLIReporter(
                metric_columns = ["loss","accuracy","training_iteration"])

        result = tune.run(partial(train_cifar,data_dir=data_dir),
                            resources_per_trial = {"cpu":2,"gpu":gpus_per_trial},
                            config = config,
                            num_samples = num_samples,
                            scheduler = scheduler,
                            progress_reporter = reporter)
        best_trial = result.get_best_trial("loss","min","last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

        best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParellel(best_trained_model)
        best_trained_model.to(device)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir,"checkpoint"))
        best_trained_model.load_state_dict(model_state)
        test_acc = test_accuracy(best_trained_model,device)
        print("Best trial test set accuracy: {}".format(test_acc))

if __name__ == "__main__":
    main(num_samples = 10,max_num_epochs = 10, gpus_per_trial = 0)

'''
Best trial config: {'l1': 8, 'l2': 16, 'lr': 0.00276249, 'batch_size': 16, 'data_dir': '...'}
Best trial final validation loss: 1.181501
Best trial final validation accuracy: 0.5836
Best trial test set accuracy: 0.5806


'''
