import torch
import torch.nn as nn
from einops import rearrange
import kornia

from modules.net import LayerNorm,FeedForward,TransformerBlock,StarBlock

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv_main = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qkv_dwconv_main = nn.Conv2d(
            dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.qkv_aux = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_dwconv_aux = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, main, aux, Rev=False):
        b, c, h, w = main.shape

        kv = self.qkv_dwconv_main(self.qkv_main(main))  # 之前是64 现在是64 * 3  [B 64 *3  h w]
        k, v = kv.chunk(2, dim=1)
        # B 64 h w  B 8 8 h*w
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = self.qkv_dwconv_aux(self.qkv_aux(aux))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        if not Rev:
            attn = attn.softmax(dim=-1)
        else:
            attn = -attn
            attn = attn.softmax(dim=-1)

        out = (attn @ v)  # c * c = c * h *w
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class ShareFuse(nn.Module):
    def __init__(self, dim=64, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type="WithBias"):
        super(ShareFuse, self).__init__()
        self.norm1_vi = LayerNorm(dim, LayerNorm_type)
        self.norm1_ir = LayerNorm(dim, LayerNorm_type)

        self.attn_vi = CrossAttention(dim, num_heads, bias)
        self.attn_ir = CrossAttention(dim, num_heads, bias)

        self.norm2_vi = LayerNorm(dim, LayerNorm_type)
        self.norm2_ir = LayerNorm(dim, LayerNorm_type)

        self.ffn_vi = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffn_ir = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, vi, ir):
        vi_norm = self.norm1_vi(vi)
        ir_norm = self.norm1_ir(ir)

        vi = vi + self.attn_vi(vi_norm, ir_norm, False)
        vi = vi + self.ffn_vi(self.norm2_vi(vi))

        ir = ir + self.attn_ir(ir_norm, vi_norm, False)
        ir = ir + self.ffn_ir(self.norm2_ir(ir))

        out = (vi + ir) / 2
        return out


class PrivateFuse(nn.Module):
    def __init__(self, dim=64, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type="WithBias"):
        super(PrivateFuse, self).__init__()
        self.norm1_vi = LayerNorm(dim, LayerNorm_type)
        self.norm1_ir = LayerNorm(dim, LayerNorm_type)

        self.attn_vi = CrossAttention(dim, num_heads, bias)
        self.attn_ir = CrossAttention(dim, num_heads, bias)

        self.norm2_vi = LayerNorm(dim, LayerNorm_type)
        self.norm2_ir = LayerNorm(dim, LayerNorm_type)

        self.ffn_vi = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffn_ir = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, vi, ir):
        vi_norm = self.norm1_vi(vi)
        ir_norm = self.norm1_ir(ir)

        vi = vi + self.attn_vi(vi_norm, ir_norm, True)
        vi = vi + self.ffn_vi(self.norm2_vi(vi))

        ir = ir + self.attn_ir(ir_norm, vi_norm, True)
        ir = ir + self.ffn_ir(self.norm2_ir(ir))
        out = torch.max(vi, ir)
        return out
    
    
    
class Stem(nn.Module):
    def __init__(self, in_c=1,embed_dim=64,bias=False) -> None:
        super().__init__()
        self.stem =  nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)


        self.share_enc = nn.Sequential(*[TransformerBlock(
                                            dim=embed_dim,
                                            num_heads=8,
                                            ffn_expansion_factor=2,
                                            bias=bias,
                                            LayerNorm_type='WithBias')
                          for _ in range(4)])
        self.private_enc = nn.Sequential(*[StarBlock(dim_in=embed_dim,dim_out=embed_dim) for _ in range(2)])

    def forward(self,img):
        x = self.stem(img)
        return self.share_enc(x),self.private_enc(x)





    
class Decoder_Tiny(nn.Module):
    def __init__(self,dim=64,bias=False,out_channels=1):
        super().__init__()
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim), kernel_size=3,stride=1,groups=int(dim), padding=1,bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()  
    
    
    def forward(self,x):
        return self.sigmoid(self.output(x))
    
class A2B(nn.Module):
    def __init__(self,dim=64,bias=False):
        super(A2B,self).__init__()
        self.vi2ir = Decoder_Tiny()
        self.ir2vi = Decoder_Tiny()
        
        self.mse  = nn.MSELoss()
        self.ssim = kornia.losses.SSIMLoss(11, reduction='mean')
        

    def test(self,vi_share,ir_share):
        fake_ir = self.vi2ir(vi_share)
        fake_vi = self.ir2vi(ir_share)
        return fake_ir,fake_vi
    def forward_loss(self,fake,real):
        return 5 * self.ssim(fake, real) + self.mse(fake, real)
    
    def forward(self,vi_share,ir_share,vi,ir):
        fake_ir = self.vi2ir(vi_share)
        fake_vi = self.ir2vi(ir_share)
        
        return self.forward_loss(fake_ir,ir) + self.forward_loss(fake_vi,vi)

    
class Encoder(nn.Module):
    def __init__(self, ):
        super(Encoder, self).__init__()
        self.enc = Stem()
    def forward(self, vi, ir):
        vi_share, vi_private = self.enc(vi)
        ir_share, ir_private = self.enc(ir)
        return vi_share, vi_private, ir_share, ir_private




class Decoder(nn.Module):
    def __init__(self,dim=64,bias=False):
        super().__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=2,
                                            bias=bias, LayerNorm_type='WithBias') for i in range(2)])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, 1, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()

    def forward(self,share_feats,private_feats):
        out_feats = torch.cat([share_feats,private_feats],dim=1)
        out_enc_level0 = self.reduce_channel(out_feats)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1)
    
    
    
class FusionMoudle(nn.Module):
    def __init__(self):
        super().__init__()
        self.share_fuse = ShareFuse()
        self.private_fuse = PrivateFuse()

    def forward(self,vi_share,vi_private,ir_share,ir_private):
        return self.share_fuse(vi_share,ir_share),self.private_fuse(vi_private,ir_private)
    

