import torch

def inference():
    net = Net()
    model = net.load_state_dict()
    model = model(torch.load(PATH))
    


if __name__ == "main":
    inference()
