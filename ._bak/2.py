from torchviz import make_dot
import torch
import torch.nn as nn
FLAG=True
class InterFuse(nn.Module):

    def __init__(self,in_dim=2,out_dim=256):
        super(InterFuse,self).__init__()
        self.conv1 = nn.Conv2d(100,256,1,1)
        self.conv2 = nn.Conv2d(256,256,1,1)
        self.conv3 = nn.Conv2d(in_dim,256,1,1)
        self.conv4 = nn.Conv2d(256,out_dim,1,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)

        nn.init.kaiming_normal_(self.conv1.weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4.weight,mode='fan_out',nonlinearity='relu')
    def forward(self,x,y):
        x1 = self.conv1(x)
        y  = self.conv3(y)
        if FLAG:
            x2 = self.bn1(x1)
            x2 = self.relu(x2)
            x2 = self.conv2(x2)
            y = self.sigmoid(y)
            out = self.relu(self.bn2(y*x2+x1))
            out = self.conv4(out)
        else:
            out = self.relu(self.bn2(y+x1))
            out = self.conv4(out)
        return out

input1 = torch.randn(1,100,104,104)
input2 = torch.randn(1,2,104,104)
model = InterFuse()
a=make_dot(model(input1,input2), params=dict(model.named_parameters()))
print(a)


