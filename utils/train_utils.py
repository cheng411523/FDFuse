import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.image_utils import Sobelxy,mean_filter,mse


def set_seed(seed):
    random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True  
    
    
class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def pixel_loss(self,vi,ir,out):
        vi_mean = mean_filter(vi,11)
        ir_mean = mean_filter(ir,11)
        zero = torch.zeros_like(vi)
        one = torch.ones_like(vi)
        mask = torch.where((vi_mean - ir_mean) > 0, one, zero)  
        loss = mask * mse(vi,out) + (1-mask) * mse(ir,out)
        loss = torch.mean(loss)
        return loss

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        loss_in = self.pixel_loss(image_y,image_ir,generate_img)
        
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad