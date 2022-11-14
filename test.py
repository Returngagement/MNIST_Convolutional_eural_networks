import torch
import torchvision
import torch.utils.data as Data
# test_data = torchvision.datasets.MNIST(root='./data/', train=False, transform=torchvision.transforms.ToTensor(), download=False)
test_data = torchvision.datasets(root='./data1/', train=False, transform=torchvision.transforms.ToTensor(), download=False)
test_loader = Data.DataLoader(test_data, batch_size=1, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net=torch.load('./ResNet18.pkl',map_location=torch.device(device))#载入模型
net=torch.load('./result/resnet_sigmoid/ResNet_sigmoid.pkl',map_location=torch.device(device))#载入模型
net.to(device)

torch.set_grad_enabled(False) #测试集不需要对模型的参数进行更新，关闭自动求导功能。
net.eval() #屏蔽Dropout层，冻结BN层的参数，防止测试阶段BN层发生参数更新

#测试模型，获取测试集的大小，再计算准确率。用.size()方法，获取其第0维度的大小
length=test_data.data.size(0)

print(length)
acc=0.0
for i,data in enumerate(test_loader):
    inputs,labels=data
    y_pred=net(inputs.to(device,torch.float))
    pred=y_pred.argmax(dim=1)
    acc+=(pred.data.cpu()==labels.data).sum()
    print('Predict:',int(pred.data.cpu()),'Ground Truth:',int(labels.data))
acc=(acc/length)*100
print('Accuracy:%.2f'%acc,'%')