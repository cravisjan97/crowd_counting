import os
import torch
import numpy as np

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils
import matplotlib.pyplot as plt
from torch.autograd import Variable

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # converts tensors to CUDA variables if gpu is available  

data_path =  './data/original/shanghaitech/part_A/test_data/images/'
gt_path = './data/original/shanghaitech/part_A/test_data/ground-truth_csv/'
model_path = './final_models/mcnn_shtechA_660.h5'

output_dir = './output_encoder/shanghai_part_A/'
model_name = os.path.basename(model_path).split('.')[0]

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


net = CrowdCounter()
      
trained_model = os.path.join(model_path)
print('Loading the best trained model ...')
network.load_net(trained_model, net)
net.to(device)
net.eval()

mae = 0.0
mse = 0.0

#load test data
data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

for it,blob in enumerate(data_loader):
    print('{}/{}'.format(it,data_loader.get_num_samples()))                        
    im_data = blob['data']
    raw_img = blob['raw_img']
    im_data = Variable(torch.from_numpy(im_data))
    enc1 = net.DME.branch1(im_data)
    enc2 = net.DME.branch2(im_data)
    enc3 = net.DME.branch3(im_data)
    
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

    f.savefig(output_dir+'output_'+blob['fname'].split('.')[0] + '.png')
    f = plt.close()
    
       


