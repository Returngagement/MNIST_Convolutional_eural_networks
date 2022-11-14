import  torch
import torch.nn as nn
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2)
        self.r1=nn.ReLU()
        self.s1=nn.MaxPool2d(kernel_size=2)
        self.c2=nn.Conv2d(6,16,5,1,0)
        self.r2=nn.ReLU()
        self.s2=nn.MaxPool2d(2)
        self.c3=nn.Conv2d(16,120,5,1,0)
        self.r3=nn.ReLU()
        self.f1=nn.Linear(in_features=120,out_features=84)
        self.r4=nn.ReLU()
        self.out=nn.Linear(84,10)
    def forward(self, x):  #x为输入的变量
        x=self.c1(x)
        x=self.r1(x)
        x=self.s1(x)
        x=self.c2(x)
        x=self.r2(x)
        x=self.s2(x)
        x=self.c3(x)
        x=self.r3(x)
        x=x.view(x.size(0),-1)
        x=self.f1(x)
        x=self.r4(x)
        x=self.out(x)
        return x
if __name__ == "__main__":
    model=LeNet()
    print(model)
    a=torch.randn(1,1,28,28)
    b=model(a)
    print(b)
    print(b.size())


