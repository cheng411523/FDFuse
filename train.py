import  random
import  numpy as np
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import kornia




from modules.build import Encoder,Decoder,FusionMoudle,A2B
from utils.train_utils import Fusionloss
from options import  TrainOptions 
from datasets import H5Datasets


parser = TrainOptions()
opts = parser.parse()

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# loss
criteria_fusion = Fusionloss()


encoder_model = nn.DataParallel(Encoder()).to(device)
decoder_model = nn.DataParallel(Decoder()).to(device)
fusion_model = nn.DataParallel(FusionMoudle()).to(device)
Temporary_model = nn.DataParallel(A2B()).to(device)
encoder_model.train()
decoder_model.train()  
fusion_model.train()
Temporary_model.train()

optimizer1 = torch.optim.Adam(encoder_model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
optimizer2 = torch.optim.Adam(decoder_model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
optimizer3 = torch.optim.Adam(fusion_model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
optimizer4 = torch.optim.Adam(Temporary_model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=opts.optim_step, gamma=opts.optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=opts.optim_step, gamma=opts.optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=opts.optim_step, gamma=opts.optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=opts.optim_step, gamma=opts.optim_gamma)



MSELoss = nn.MSELoss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')

trainloader = DataLoader(H5Datasets(opts.train_data),batch_size=opts.batch_size,shuffle=True,num_workers=0)
loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
prev_time = time.time()
for epoch in range(opts.total_epoch):
    for i, (vi, ir) in enumerate(loader['train']):
        vi, ir = vi.cuda(), ir.cuda()
        encoder_model.zero_grad()
        decoder_model.zero_grad()
        fusion_model.zero_grad()
        Temporary_model.zero_grad()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        if epoch < opts.gap_epoch: 
            vi_share,vi_private,ir_share,ir_private = encoder_model(vi,ir)
            loss_decomp = Temporary_model(vi_share,ir_share,vi,ir)
            vi_hat,ir_hat = decoder_model(vi_share,vi_private),decoder_model(ir_share,ir_private)
            loss_recon = 5 * Loss_ssim(vi, vi_hat) + MSELoss(vi, vi_hat) + 5 * Loss_ssim(ir, ir_hat) + MSELoss(ir, ir_hat)
            loss =   5 * loss_decomp + 2 * loss_recon   
            loss.backward()

            nn.utils.clip_grad_norm_(encoder_model.parameters(), max_norm=0.01, norm_type=2)
            nn.utils.clip_grad_norm_(Temporary_model.parameters(), max_norm=0.01, norm_type=2)
            nn.utils.clip_grad_norm_(decoder_model.parameters(), max_norm=0.01, norm_type=2)

            optimizer1.step()  
            optimizer2.step()
            optimizer4.step()
            
        else:  
            vi_share,vi_private,ir_share,ir_private = encoder_model(vi,ir)
            fuse_share,fuse_private = fusion_model(vi_share,vi_private,ir_share,ir_private)
            out = decoder_model(fuse_share,fuse_private)
            loss,_,_  = criteria_fusion(vi, ir, out)
            loss.backward()
                    
            nn.utils.clip_grad_norm_(encoder_model.parameters(), max_norm=0.01, norm_type=2)
            nn.utils.clip_grad_norm_(decoder_model.parameters(), max_norm=0.01, norm_type=2)
            nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=0.01, norm_type=2)
            nn.utils.clip_grad_norm_(Temporary_model.parameters(), max_norm=0.01, norm_type=2)
            optimizer1.step()  
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
        batches_done = epoch * len(loader['train']) + i
        batches_left = opts.total_epoch * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write( 
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f]  ETA: %.10s"
            % (
                epoch,
                opts.total_epoch,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )
        
    scheduler1.step()  
    scheduler2.step()
    scheduler4.step()
    if not epoch < opts.gap_epoch:
        scheduler3.step()
        

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6

checkpoint = {
    'encoder': encoder_model.state_dict(),
    'decoder': decoder_model.state_dict(),
    'fuse': fusion_model.state_dict(),
}
torch.save(checkpoint,"FDFuse_2024.pth")
