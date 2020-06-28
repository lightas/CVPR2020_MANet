import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
#####
class _res_block(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(_res_block,self).__init__()
        self.conv1=nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1,groups=out_dim)
        self.relu1=nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2=nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1,groups=out_dim)
        self.relu2=nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_dim)
    def forward(self,x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x += res
        return x


####################
class IntSegHead(nn.Module):
    def __init__(self,in_dim=103,emb_dim=256):
        super(IntSegHead,self).__init__()
#        self.conv1 = nn.Conv2d(in_dim,out_dim,kernel_size=7,stride=1,padding=3,groups=in_dim)
#        self.bn1 = nn.BatchNorm2d(emb_dim)
#        self.relu1 = nn.ReLU()
        self.layer1 = _split_separable_conv2d(in_dim,emb_dim)
        self.res1 = _res_block(emb_dim,emb_dim)
        self.res2 = _res_block(emb_dim,emb_dim)
        self.conv2 = nn.Conv2d(256,emb_dim,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(emb_dim)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(emb_dim,1,1,1)
    def forward(self,x):
#        x = self.conv1(x)
#        x = self.bn1(x)
#        x = self.relu1(x)
        x = self.layer1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x
#####
class _split_separable_conv2d(nn.Module):
    def __init__(self,in_dim,out_dim,kernel_size=7):
        super(_split_separable_conv2d,self).__init__()
        self.conv1=nn.Conv2d(in_dim,in_dim,kernel_size=kernel_size,stride=1,padding=int((kernel_size-1)/2),groups=in_dim)
        self.relu1=nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv2=nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1)
        self.relu2=nn.ReLU(True)
        self.bn2 = nn.BatchNorm2d(out_dim)
        nn.init.kaiming_normal_(self.conv1.weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight,mode='fan_out',nonlinearity='relu')
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class DynamicSegHead(nn.Module):
    def __init__(self,in_dim=103,embed_dim=256,kernel_size=1):
        super(DynamicSegHead,self).__init__()
        self.layer1=_split_separable_conv2d(in_dim,embed_dim)
        self.layer2=_split_separable_conv2d(embed_dim,embed_dim)
        self.layer3=_split_separable_conv2d(embed_dim,embed_dim)
        self.layer4=_split_separable_conv2d(embed_dim,embed_dim)
        self.conv=nn.Conv2d(embed_dim,1,1,1)
        nn.init.kaiming_normal_(self.conv.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv(x)
        return x
#####
with torch.cuda.device(0):
    net = DynamicSegHead()
    flops, params = get_model_complexity_info(net, (103, 104, 104), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)
