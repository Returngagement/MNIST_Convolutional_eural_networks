import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from resnet_sigmoid import ResNet18


model =ResNet18()

Epoch=20
batch_size=64
lr=0.001
train_data=torchvision.datasets.MNIST(root='./data/',train=True,transform=torchvision.transforms.ToTensor(),download=False)
train_loader=Data.DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr) #传入模型的参数

torch.set_grad_enabled(True) #启动pytorch的自动求导机制
model.train()  #启用batch normalization和dropout

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
length=train_data.data.size(0)
print(length)

with open("./result/resnet_sigmoid/resnet_sigmoid.txt", "w") as f:
    for epoch in range(Epoch):
        running_loss=0.0
        acc=0.0
        for step,data in enumerate(train_loader,0):
            x, y = data  # x为inputs输入的数据， y为labels标签
            optimizer.zero_grad()  # 梯度清零
            y_pred = model(x.to(device, torch.float))  # 训练输出的y_pred
            loss = loss_function(y_pred, y.to(device, torch.long))
            loss.backward()  # 损失值反向传播
            running_loss+=float(loss.data.cpu())#loss传回cpu并且转换为float类型
            pred = y_pred.argmax(dim=1)#y_pred是一个二维的张量，其形状为[batch_size, channel]，在这边channel是10，即十个数字。
            # 如果我们将batch中的任意一行提取出来就获得了一个10维的向量，向量里的每个数代表与其下标所对应的标签的相关性，相关性越大则代表越有可能是这个数字。因此，我们需要获得这个向量中最大数的下标，在pytorch中，我们可以用.argmax(dim)方法实现，输入维度dim，即可返回这个维度下最大值的下标，即pred = y_pred.argmax(dim=1)。
            acc+=(pred.data.cpu()==y.data).sum()
            optimizer.step()  # 对参数进行更新
            if step % 100 == 99:
                loss_avg=running_loss/(step+1)
                acc_avg=float(acc/((step+1)*batch_size))
                print('Epoch',epoch+1,'step',step+1,'|Loss_avg:%4f'%loss_avg,'|Acc_avg:%.4f'%acc_avg)
                f.write('Epoch: %03d  step: %03d |Loss_avg: %.4f | Acc_avg: %.4f%% '
                        % (epoch + 1, step + 1, loss_avg, acc_avg))
                f.write('\n')
                f.flush()
torch.save(model,'./result/resnet_sigmoid/ResNet_sigmoid.pkl')