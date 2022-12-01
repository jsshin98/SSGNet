import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from model.network import MMSR_net   
from PIL import Image

DEFAULT_PARAMS = {
                  'scale': -1,
  
                  'loss': 'l1',
                  'optim': 'adam',
                  'lr': -1,
                  'weights_regularizer': [-1, -1, -1, -1], 
    
                  'batch_size': -1,                            
                  'epoch': -1,                              
                  }                                     

                            
def MMSR(guide_img, source_img, mask=None, params=DEFAULT_PARAMS, target_img=None, index=1, max_v=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if mask is not None:
        mask = mask.unsqueeze(0)
    source_img = source_img.squeeze()                                                                                         
    r = params['scale']                                        
    # normalization
    guide_img = (guide_img - np.mean(guide_img, axis=(1, 2), keepdims=True)) / np.std(guide_img, axis=(1, 2), keepdims=True)
                                                                                     
    source_img_mean = np.mean(source_img)
    source_img_std = np.std(source_img)
    source_img = (source_img - source_img_mean) / source_img_std
    
     # prepare data
    guide_img = torch.from_numpy(guide_img).float().to(device)
    source_img = torch.from_numpy(source_img).float().to(device)
    guide_for_opt = guide_img.unsqueeze(0)                                        
    source_for_opt = source_img.unsqueeze(0).unsqueeze(0)
    if target_img is not None:
        target_img = torch.from_numpy(target_img).float().to(device)

    train_data = torch.utils.data.TensorDataset(guide_for_opt, source_for_opt)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'])     
    
    # setup network                                                            
    mynetwork = MMSR_net().train().to(device)
    optimizer = optim.Adam(mynetwork.params_with_regularizer, lr=params['lr'])
    print('# of eigen network parameter:', sum(p.numel() for p in mynetwork.parameters() if p.requires_grad))
    if params['loss'] == 'l1':
        myloss = torch.nn.L1Loss()
    elif params['loss'] == 'mse':                                       
        myloss = torch.nn.MSELoss()
    else:
        print("unknown loss!")
        return

    # train network                                                           
    for epoch in range(params['epoch']):
        if (epoch+1) % 5 == 0:                                               
            for p in optimizer.param_groups:
                p['lr'] *= 0.9998
        mynetwork.train()

        for (x, y) in train_loader:
            optimizer.zero_grad()                                                

            pred,_ = mynetwork(x, y)
            pred_down = F.avg_pool2d(pred, r)
            if mask is not None:
                consistency = myloss(pred_down[mask==1.0], y[mask==1.0])
            else:
                consistency = myloss(pred_down, y)
            consistency.backward()
            optimizer.step()
    # final prediction
    mynetwork.eval()         
    prediction, eigen_vs = mynetwork(guide_img.unsqueeze(0), source_img.unsqueeze(0).unsqueeze(0))
    prediction = source_img_mean + source_img_std * prediction.squeeze()  # denormalization
    prediction = prediction.cpu().detach().squeeze().numpy()

    return prediction, eigen_vs
