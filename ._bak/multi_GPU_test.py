import torch
import torch.nn as nn

#
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(10,5)
        self.fc2 = nn.Linear(5,5)
        self.fc3 = nn.Linear(5,10)
    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

inputs = torch.rand(14,10)
net=Net()
net = net.cuda()
net.train()
net = nn.DataParallel(net)
f = net(inputs)
print(f.size())
net2 = Net()
net2 = net.cuda()
net2.train()
net2 =nn.DataParallel(net2)
dic ={}
for i in range(f.size(0)):
    if i<3:
        f[i].to(torch.device('cuda:0'))
        
    else:
        f[i].to(torch.device('cuda:1'))
    dic[i]=net2(f[i].unsqueeze(0))
print(dic)
gt = torch.ones(14)
for i in range(10):
    gt[i]=i



print(gt)
