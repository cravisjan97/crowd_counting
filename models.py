import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=False):
        super(ConvBlock, self).__init__()
        padding = int((kernel_size - 1) / 2) 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) 

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x

class MCNN(nn.Module):
    '''
    Multi-column CNN 
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    
    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        
        self.branch1 = nn.Sequential(ConvBlock( 1, 16, 9, bn=bn),
                                     nn.MaxPool2d(2),
                                     ConvBlock(16, 32, 7, bn=bn),
                                     nn.MaxPool2d(2),
                                     ConvBlock(32, 16, 7, bn=bn),
                                     ConvBlock(16,  8, 7, bn=bn))
        
        self.branch2 = nn.Sequential(ConvBlock( 1, 20, 7, bn=bn),
                                     nn.MaxPool2d(2),
                                     ConvBlock(20, 40, 5, bn=bn),
                                     nn.MaxPool2d(2),
                                     ConvBlock(40, 20, 5, bn=bn),
                                     ConvBlock(20, 10, 5, bn=bn))
        
        self.branch3 = nn.Sequential(ConvBlock(1, 24, 5, bn=bn),
                                     nn.MaxPool2d(2),
                                     ConvBlock(24, 48, 3, bn=bn),
                                     nn.MaxPool2d(2),
                                     ConvBlock(48, 24, 3, bn=bn),
                                     ConvBlock(24, 12, 3, bn=bn))
        
        self.fuse = nn.Sequential(ConvBlock( 30, 1, 1, bn=bn))
        
    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1,x2,x3),1)
        x = self.fuse(x)
        
        return x

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                #print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)
