import os
import numpy as np
import torch
import torch.nn as nn
from skimage.io import imsave
from modules.build import Encoder,Decoder,FusionMoudle
from utils.image_utils import image_read_cv2
from options import  TrainOptions 


def test(ckpt,vi_path,ir_path,out_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = nn.DataParallel(Encoder()).to(device)
    encoder.load_state_dict(torch.load(ckpt,weights_only=True)['encoder'])
    encoder.eval()
    
    decoder = nn.DataParallel(Decoder()).to(device) 
    decoder.load_state_dict(torch.load(ckpt,weights_only=True)['decoder'])
    decoder.eval()
    
    fuser  = nn.DataParallel(FusionMoudle()).to(device)
    fuser.load_state_dict(torch.load(ckpt,weights_only=True)['fuse'])
    fuser.eval()

    with torch.no_grad():
        for img_name in os.listdir(vi_path):
            Img_vi = image_read_cv2(os.path.join(vi_path, img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            Img_ir = image_read_cv2(os.path.join(ir_path, img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            Img_vi = torch.FloatTensor(Img_vi)
            Img_ir = torch.FloatTensor(Img_ir)
            Img_vi, Img_ir = Img_vi.cuda(), Img_ir.cuda()
            vi_share, vi_private, ir_share, ir_private = encoder(Img_vi,Img_ir)
            feats_share, feats_private = fuser(vi_share, vi_private, ir_share, ir_private)
            out = decoder(feats_share, feats_private)
            out = (out - torch.min(out)) / (torch.max(out) - torch.min(out))
            out = (out * 255.0).cpu().numpy().squeeze(0).squeeze(0).astype('uint8') 
            imsave(os.path.join(out_path, "{}.png".format(img_name.split(sep='.')[0])),out)



if __name__ == "__main__":
    parser = TrainOptions()
    opts = parser.parse()
    test(opts.ckpt_path,opts.vi_path,opts.ir_path,opts.out_path)    