import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', bn=False):
        super(ConvBlock, self).__init__()
        padding = int((kernel_size - 1) / 2) 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU() 
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FC(nn.Module):
    def __init__(self, in_features, out_features, NL='relu'):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU() 
        else:
            self.relu = None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CMTL(nn.Module):
    '''
    Implementation of CNN-based Cascaded Multi-task Learning of High-level Prior and Density
    Estimation for Crowd Counting (Sindagi et al.)
    '''
    
    def __init__(self, bn=False, num_classes=10):
        super(CMTL, self).__init__()
        
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(ConvBlock( 1, 16, 9, NL='prelu', bn=bn),                                     
                                        ConvBlock(16, 32, 7, NL='prelu', bn=bn))
        
        self.hl_prior_1 = nn.Sequential(ConvBlock( 32, 16, 9, NL='prelu', bn=bn),
                                     nn.MaxPool2d(2),
                                     ConvBlock(16, 32, 7, NL='prelu', bn=bn),
                                     nn.MaxPool2d(2),
                                     ConvBlock(32, 16, 7, NL='prelu', bn=bn),
                                     ConvBlock(16, 8,  7, NL='prelu', bn=bn))
                
        self.hl_prior_2 = nn.Sequential(nn.AdaptiveMaxPool2d((32,32)),
                                        ConvBlock( 8, 4, 1, NL='prelu', bn=bn))
        
        self.hl_prior_fc1 = FC(4*1024,512, NL='prelu')
        self.hl_prior_fc2 = FC(512,256, NL='prelu')
        self.hl_prior_fc3 = FC(256, self.num_classes, NL='prelu')
        
        
        self.de_stage_1 = nn.Sequential(ConvBlock( 32, 20, 7, NL='prelu', bn=bn),
                                     nn.MaxPool2d(2),
                                     ConvBlock(20, 40, 5, NL='prelu', bn=bn),
                                     nn.MaxPool2d(2),
                                     ConvBlock(40, 20, 5, NL='prelu', bn=bn),
                                     ConvBlock(20, 10, 5, NL='prelu', bn=bn))
        
        self.de_stage_2 = nn.Sequential(ConvBlock( 18, 24, 3, NL='prelu', bn=bn),
                                        ConvBlock( 24, 32, 3, NL='prelu', bn=bn),                                        
                                        nn.ConvTranspose2d(32,16,4,stride=2,padding=1,output_padding=0,bias=True),
                                        nn.PReLU(),
                                        nn.ConvTranspose2d(16,8,4,stride=2,padding=1,output_padding=0,bias=True),
                                        nn.PReLU(),
                                        ConvBlock(8, 1, 1, NL='relu', bn=bn))        
    def forward(self, im_data):
        x_base = self.base_layer(im_data)
        x_hlp1 = self.hl_prior_1(x_base)
        x_hlp2 = self.hl_prior_2(x_hlp1)
        x_hlp2 = x_hlp2.view(x_hlp2.size()[0], -1) 
        x_hlp = self.hl_prior_fc1(x_hlp2)
        x_hlp = F.dropout(x_hlp, training=self.training)
        x_hlp = self.hl_prior_fc2(x_hlp)
        x_hlp = F.dropout(x_hlp, training=self.training)
        x_cls = self.hl_prior_fc3(x_hlp)        
        x_den = self.de_stage_1(x_base)        
        x_den = torch.cat((x_hlp1,x_den),1)
        x_den = self.de_stage_2(x_den)
        return x_den, x_cls


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
