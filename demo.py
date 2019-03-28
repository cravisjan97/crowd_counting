import os
import torch
import numpy as np

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # converts tensors to CUDA variables if gpu is available  

data_path =  './data/original/shanghaitech/part_A/test_data/images/'
gt_path = './data/original/shanghaitech/part_A/test_data/ground-truth_csv/'
model_path = './final_models/mcnn_shtechA_660.h5'

output_dir = './output_demo/shanghai_part_A/'
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
    gt_data = blob['gt_density']
    raw_img = blob['raw_img']
    density_map = net(im_data, gt_data)
    density_map = density_map.data.cpu().numpy()
    #print(np.shape(im_data))
    gt_count = np.sum(gt_data)
    et_count = np.sum(density_map)
    mae += abs(gt_count-et_count)
    mse += ((gt_count-et_count)*(gt_count-et_count))
 
    
    f = plt.figure()
    ax1 = f.add_subplot(1,3,1)
    ax2 = f.add_subplot(1,3,2)
    ax3 = f.add_subplot(1,3,3)

    ax1.imshow(raw_img)
    ax1.axis('off')
    ax1.text(0.5,-0.2,'Test Image',ha='center',transform=ax1.transAxes)

    ax2.imshow(gt_data[0][0])
    ax2.axis('off')
    ax2.set_title('GT Count: '+str(int(gt_count)))
    ax2.text(0.5,-0.2,'GT Density Map',ha='center',transform=ax2.transAxes)

    ax3.imshow(density_map[0][0])
    ax3.axis('off')
    ax3.set_title('Pred Count: '+str(int(et_count)))
    ax3.text(0.5,-0.2,'Predicted Density Map',ha='center',transform=ax3.transAxes)
   

    f.savefig(output_dir+'output_'+blob['fname'].split('.')[0] + '.png')
    f = plt.close()
    
    #if save_output:
    #    utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')
        
mae = mae/data_loader.get_num_samples()
mse = np.sqrt(mse/data_loader.get_num_samples())
print '\nMAE: %0.2f, MSE: %0.2f' % (mae,mse)


