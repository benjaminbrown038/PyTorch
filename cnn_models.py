model = nn.Sequential(
			nn.Conv2d(1,32,kernel_size = 3),
			nn.Conv2d(32,64,kernel_szie = 3),
			nn.MaxPool2d(2),
			nn.Dropout(0.25),
			nn.Flatten(),
			nn.Linear(9216,128),
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
