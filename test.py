import os
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import utils
from models import MCNN
from shanghai_loader import Shanghai_Dataset



torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # converts tensors to CUDA variables if gpu is available  

data_path =  './data/original/shanghaitech/part_A/test_data/images/'
gt_path = './data/original/shanghaitech/part_A/test_data/ground-truth_csv/'
model_path = './saved_models/mcnn_shtechA_1.pth'

output_dir = './output/'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


net = MCNN()
      
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
    gt_data = blob['gt_density'][0]
    density_map = net(im_data)
    density_map = density_map.data.cpu().numpy()
    gt_data = gt_data.data.cpu().numpy()
    gt_count = np.sum(gt_data)
    et_count = np.sum(density_map)
    mae += abs(gt_count-et_count)
    mse += ((gt_count-et_count)*(gt_count-et_count))
    if vis:
        utils.display_results(im_data, gt_data, density_map)
    if save_output:
        utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'][0].split('.')[0] + '.png')
        
mae = mae/len(test_dataset)
mse = np.sqrt(mse/len(test_dataset))
print '\nMAE: %0.2f, MSE: %0.2f' % (mae,mse)

f = open(file_results, 'w') 
f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
f.close()
