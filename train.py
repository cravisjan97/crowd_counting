import os
import torch
import numpy as np
import sys
import time
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models import MCNN, weights_normal_init
from shanghai_loader import Shanghai_Dataset

method = 'mcnn'
dataset_name = 'shtechA'
checkpoint_dir = './saved_models/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train'
train_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
val_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val'
val_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val_den'

#training configuration
num_epochs=2000
lr = 0.00001
momentum = 0.9

# load net
net = MCNN()
weights_normal_init(net, dev=0.01)
net.to(device)
net.train()

optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.MSELoss()

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

print('Loading training and validation datasets')
#train_data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
#val_data_loader = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)
train_dataset = Shanghai_Dataset(train_path, train_gt_path, gt_downsample=True, pre_load=True)
train_data_loader =  DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
val_dataset = Shanghai_Dataset(val_path, val_gt_path, gt_downsample=True, pre_load=True)
val_data_loader =  DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
best_mae = sys.maxint

def evaluate_model(checkpoint_path, data_loader):
    print('Evaluation starts!!!')
    net = MCNN()
    net.load_state_dict(torch.load(checkpoint_path))
    net.to(device)
    net.eval()
    mae = 0.0
    mse = 0.0
    for blob in data_loader:                        
        im_data = blob['data'][0]
        gt_data = blob['gt_density'][0]
        #im_data = Variable(torch.from_numpy(im_data))
        density_map = net(im_data)
        density_map = density_map.data.cpu().numpy()
        gt_data = gt_data.cpu().numpy()
        gt_count = np.sum(gt_data)
        et_count = np.sum(density_map)
        mae += abs(gt_count-et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))        
    mae = mae/len(val_dataset)
    mse = np.sqrt(mse/len(val_dataset))
    return mae,mse

tr_losses=[]
print('Training started!!!')
for epoch in range(num_epochs):    
    iter_cnt = 0
    train_loss = 0.0
    start_time=time.time()
    for blob in train_data_loader:                
        im_data = blob['data'][0]
        gt_data = blob['gt_density'][0]
        #im_data = Variable(torch.from_numpy(im_data))
        #gt_data = Variable(torch.from_numpy(gt_data))
        density_map = net(im_data)
        loss = criterion(density_map, gt_data)
        train_loss += loss.data[0]
        iter_cnt += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iter_cnt % 500 ==0):
            print('Iteration:{} Loss:{} Time:{}m {}s'.format(iter_cnt,loss.data[0], (time.time()-start_time)//60, (time.time()-start_time)%60))
    end_time = time.time() - start_time
    print('Epoch:{}/{}  Loss:{}  Time:{}m {}s'.format(epoch+1,num_epochs,train_loss/iter_cnt, end_time//60, end_time%60))
    tr_losses.append(train_loss/iter_cnt)
    if (epoch % 2 == 0):
        torch.save(net.state_dict(), checkpoint_dir+method+'_'+dataset_name+'_'+str(epoch+1)+'.pth')      

        #calculate error on the validation dataset 
        mae,mse = evaluate_model(checkpoint_dir+method+'_'+dataset_name+'_'+str(epoch+1)+'.pth', val_data_loader)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_model = checkpoint_dir+method+'_'+dataset_name+'_'+str(epoch+1)+'.pth'
            print('Best MSE:{}  Best MAE:{}  Best MODEL:{}'.format(best_mse,best_mae,best_model))

tr_losses = np.array(tr_losses)
np.save('Train_losses_'+dataset_name+'.npy',tr_losses)

