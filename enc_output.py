import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import MCNN
from shanghai_loader import Shanghai_Dataset

import matplotlib.pyplot as plt
from torch.autograd import Variable

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # converts tensors to CUDA variables if gpu is available  

data_path =  './data/original/shanghaitech/part_A/test_data/images/'
gt_path = './data/original/shanghaitech/part_A/test_data/ground-truth_csv/'
model_path = './saved_models/mcnn_shtechA_1.pth'

output_dir = './output_encoder/shanghai_part_A/'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


net = MCNN()
      
print('Loading the best trained model ...')
net.load_state_dict(torch.load(model_path))
net.to(device)
net.eval()

mae = 0.0
mse = 0.0

#load test data
test_dataset = Shanghai_Dataset(data_path, gt_path, gt_downsample=True, pre_load=True)
test_data_loader =  DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

for it,blob in enumerate(test_data_loader):
    print('{}/{}'.format(it,len(test_dataset)))                        
    im_data = blob['data'][0]
    raw_img = blob['raw_img'][0]
    enc1 = net.branch1(im_data)
    enc2 = net.branch2(im_data)
    enc3 = net.branch3(im_data)
    
    enc1 = enc1.detach().numpy()
    enc2 = enc2.detach().numpy()
    enc3 = enc3.detach().numpy()
    
    f = plt.figure()
    ax1 = f.add_subplot(3,2,3)
    ax2 = f.add_subplot(3,2,2)
    ax3 = f.add_subplot(3,2,4)
    ax4 = f.add_subplot(3,2,6)

    ax1.imshow(raw_img)
    ax1.axis('off')
    ax1.text(0.5,-0.1,'Test Image',ha='center',transform=ax1.transAxes)

    ax2.imshow(np.mean(enc1,axis=(0,1)))
    ax2.axis('off')
    ax2.text(0.5,-0.1,'Average Channel 1 Feature Map',ha='center',transform=ax2.transAxes)

    ax3.imshow(np.mean(enc2,axis=(0,1)))
    ax3.axis('off')
    ax3.text(0.5,-0.1,'Average Channel 2 Feature Map',ha='center',transform=ax3.transAxes)
   
    ax4.imshow(np.mean(enc3,axis=(0,1)))
    ax4.axis('off')
    ax4.text(0.5,-0.1,'Average Channel 3 Feature Map',ha='center',transform=ax4.transAxes)

    f.savefig(output_dir+'output_'+blob['fname'][0].split('.')[0] + '.png')
    f = plt.close()
    
       


