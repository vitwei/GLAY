from functools import partial
import math
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, LayerNorm2d, trunc_normal_
from einops import rearrange, repeat
from src.DECconv import DEBlock
import ipdb
_glnet_ckpt_urls= {
    'glnet_4g': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5RzlEalNfQU1xbkpRaVgxP2U9dE9raEhQ/root/content',
    'glnet_9g': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5RUdGUTZrZldWLXdWWmVpP2U9d1d3Ujh6/root/content',
    'glnet_16g': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5amFCeEMzaC1COENIV01tP2U9R2ZSMGtn/root/content',
    'glnet_stl': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5QkZhQUlMRU11X2R0YmJWP2U9OUdoaGkz/root/content',
    'glnet_stl_paramslot': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5RzBlYXQ1ekNFOXVwY1FSP2U9dm1ibW8x/root/content',
    'glnet_4g_tklb': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5QVdjT2RsVFVhMkozdnNYP2U9d0U0Y2gx/root/content',
    'glnet_9g_tklb': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5blBFY2ctZkM3WkRDTUV0P2U9Ynd1ZmVS/root/content',
}

def _load_from_url(model:nn.Module, ckpt_key:str, state_dict_key:str='model'):
    url = _glnet_ckpt_urls[ckpt_key]
    checkpoint = torch.hub.load_state_dict_from_url(url=url,
        map_location="cpu", check_hash=True, file_name=f"{ckpt_key}.pth")
    model.load_state_dict(checkpoint[state_dict_key])
    return model

class ResDWConvNCHW(nn.Conv2d):
    def __init__(self, dim, ks:int=3) -> None:
        super().__init__(dim, dim, ks, 1, padding=ks//2, bias=True, groups=dim)

    def forward(self, x:torch.Tensor):
        res = super().forward(x)
        return x + res

class LayerScale(nn.Module):
    def __init__(self, chans, init_value=1e-4, in_format='nlc') -> None:
        super().__init__()
        assert in_format in {'nlc', 'nchw'}
        if in_format == 'nlc':
            self.gamma = nn.Parameter(torch.ones((chans))*init_value, requires_grad=True)
        else: # nchw
            self.gamma = nn.Parameter(torch.ones((1, chans, 1, 1))*init_value, requires_grad=True)

    def forward(self, x:torch.Tensor):
        return self.gamma * x

class MHSA_Block(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, dropout=0.,
                 mlp_ratio:float=4., drop_path:float=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 layerscale=-1) -> None:
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.mha_op = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(mlp_ratio*embed_dim)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio*embed_dim), embed_dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ls1 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nlc') if layerscale > 0 else nn.Identity()
        self.ls2 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nlc') if layerscale > 0 else nn.Identity()

    def forward(self, x:torch.Tensor):
        """
        args:
            x: (bs, len, c) Tensor
        return:
            (bs, len, c) Tensor
        """
        # print(x.dtype)
        shortcut = x
        x = self.norm1(x)
        # print(x.dtype) -> float32 -> RuntimeError: expected scalar type Half but found Float
        # FIXME: below is just a workaround
        # https://github.com/NVIDIA/apex/issues/121#issuecomment-1235109690
        if not self.training:
            x = x.to(shortcut.dtype)

        x, attn_weights = self.mha_op(query=x, key=x, value=x, need_weights=False)
        x = shortcut + self.drop_path(self.ls1(x))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x



class GLMixBlock(nn.Module):
    """
    multihead attention with soft grouping + MLP
    """
    def __init__(self,
        embed_dim:int,
        num_heads:int,
        num_slots:int=64,
        slot_init:str='ada_avgpool',
        slot_scale:float=None,
        scale_mode='learnable',
        local_dw_ks:int=5,
        mlp_ratio:float=4.,
        drop_path:float=0.,
        norm_layer=LayerNorm2d,
        cpe_ks:int=0, # if > 0, the kernel size of the conv pos embedding
        mlp_dw:bool=False,
        layerscale=-1,
        use_slot_attention:bool=True,
        ) -> None:

        super().__init__()
        self.embed_dim = embed_dim
        slot_scale = slot_scale or embed_dim ** (-0.5)
        self.scale_mode = scale_mode
        self.use_slot_attention = use_slot_attention
        assert scale_mode in {'learnable', 'const'}
        if scale_mode  == 'learnable':
            self.slot_scale = nn.Parameter(torch.tensor(slot_scale))
        else: # const
            self.register_buffer('slot_scale', torch.tensor(slot_scale))

        # convolutional position encoding
        self.with_conv_pos_emb = (cpe_ks > 0)
        if self.with_conv_pos_emb:
            self.pos_conv = nn.Conv2d(
                embed_dim, embed_dim,
                kernel_size=cpe_ks,
                padding=cpe_ks//2, groups=embed_dim)

        # slot initialization
        assert slot_init in {'param', 'ada_avgpool','ada_maxpool'}
        self.slot_init = slot_init
        if self.slot_init == 'param':
            self.init_slots = nn.Parameter(
                torch.empty(1, num_slots, embed_dim), True)
            torch.nn.init.normal_(self.init_slots, std=.02)
        else:
            self.pool_size = math.isqrt(num_slots)
            # TODO: relax the square number constraint
            assert self.pool_size**2 == num_slots
        
        # spatial mixing
        self.norm1 = norm_layer(embed_dim)
        self.relation_mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True) if use_slot_attention else nn.Identity()
        self.feature_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1), # pseudo qkv linear
            nn.Conv2d(embed_dim, embed_dim, local_dw_ks, padding=local_dw_ks//2, groups=embed_dim), # pseudo attention
            nn.Conv2d(embed_dim, embed_dim, 1), # pseudo out linear
        ) if local_dw_ks > 0 else nn.Identity()
        
        # per-location embedding
        self.norm2 = norm_layer(embed_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, int(mlp_ratio*embed_dim), kernel_size=1),
            ResDWConvNCHW(int(mlp_ratio*embed_dim),ks=3) if mlp_dw else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(int(mlp_ratio*embed_dim), embed_dim, kernel_size=1)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # layer scale
        self.ls1 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nchw') \
            if layerscale > 0 else nn.Identity()
        self.ls2 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nchw') \
            if layerscale > 0 else nn.Identity()

        # NOTE: hack, for visualization with forward hook
        self.vis_proxy = nn.Identity()
        
    def _forward_relation(self, x:torch.Tensor, init_slots:torch.Tensor):
        """
        x: (bs, c, h, w) tensor
        init_slots: (bs, num_slots, c) tensor
        """
        x_flatten = x.permute(0, 2, 3, 1).flatten(1, 2)
        logits = F.normalize(init_slots, p=2, dim=-1) @ \
            (self.slot_scale*F.normalize(x_flatten, p=2, dim=-1).transpose(-1, -2)) # (bs, num_slots, l)
        # soft grouping
        slots = torch.softmax(logits, dim=-1) @ x_flatten # (bs, num_slots, c)
        # cluster update with mha
        slots, attn_weights = self.relation_mha(query=slots, key=slots, value=slots, need_weights=False)

        if not self.training: # hack, for visualization
            logits, attn_weights = self.vis_proxy((logits, attn_weights))

        # soft ungrouping
        out = torch.softmax(logits.transpose(-1, -2), dim=-1) @ slots # (bs, h*w, c)
    
        out = out.permute(0, 2, 1).reshape_as(x) # (b, c, h, w)
        # # fuse with locally enhanced features
        out = out + self.feature_conv(x)

        return out, slots

    def forward(self, x:torch.Tensor):
        """
        x: (bs, c, h, w) tensor
        init_slots: (bs, num_slots, c) tensor
        """
        if self.slot_init == 'ada_avgpool':
            init_slots = F.adaptive_avg_pool2d(x, output_size=self.pool_size
                ).permute(0, 2, 3, 1).flatten(1, 2)
        elif self.slot_init == 'ada_maxpool':
            init_slots = F.adaptive_max_pool2d(x, output_size=self.pool_size
                ).permute(0, 2, 3, 1).flatten(1, 2)
        else:
            init_slots = self.init_slots

        # print(x.dtype, init_slots.dtype) -> float16 float32
        if self.with_conv_pos_emb:
            x = x + self.pos_conv(x)

        shortcut = x
        x, updt_slots = self._forward_relation(self.norm1(x), init_slots)
        x = shortcut + self.drop_path(self.ls1(x))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        return x
    
    def extra_repr(self) -> str:
        return f"scale_mode={self.scale_mode}, "\
               f"slot_init={self.slot_init}, "\
               f"use_slot_attention={self.use_slot_attention}"

class myGLMixBlock(nn.Module):
    """
    multihead attention with soft grouping + MLP
    """
    def __init__(self,
        embed_dim:int,
        num_heads:int,
        num_slots:int=64,
        slot_init:str='ada_avgpool',
        slot_scale:float=None,
        scale_mode='learnable',
        local_dw_ks:int=5,
        mlp_ratio:float=4.,
        drop_path:float=0.,
        norm_layer=LayerNorm2d,
        cpe_ks:int=0, # if > 0, the kernel size of the conv pos embedding
        mlp_dw:bool=False,
        layerscale=-1,
        use_slot_attention:bool=True,
        ) -> None:

        super().__init__()
        self.embed_dim = embed_dim
        slot_scale = slot_scale or embed_dim ** (-0.5)
        self.scale_mode = scale_mode
        self.use_slot_attention = use_slot_attention
        assert scale_mode in {'learnable', 'const'}
        if scale_mode  == 'learnable':
            self.slot_scale = nn.Parameter(torch.tensor(slot_scale))
        else: # const
            self.register_buffer('slot_scale', torch.tensor(slot_scale))

        # convolutional position encoding
        self.with_conv_pos_emb = (cpe_ks > 0)
        if self.with_conv_pos_emb:
            self.pos_conv = nn.Conv2d(
                embed_dim, embed_dim,
                kernel_size=cpe_ks,
                padding=cpe_ks//2, groups=embed_dim)

        # slot initialization
        assert slot_init in {'param', 'ada_avgpool','ada_maxpool'}
        self.slot_init = slot_init
        if self.slot_init == 'param':
            self.init_slots = nn.Parameter(
                torch.empty(1, num_slots, embed_dim), True)
            torch.nn.init.normal_(self.init_slots, std=.02)
        else:
            self.pool_size = math.isqrt(num_slots)
            # TODO: relax the square number constraint
            assert self.pool_size**2 == num_slots
        
        # spatial mixing
        self.norm1 = norm_layer(embed_dim)
        self.relation_mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True) if use_slot_attention else nn.Identity()
        self.feature_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1), # pseudo qkv linear
            nn.Conv2d(embed_dim, embed_dim, local_dw_ks, padding=local_dw_ks//2, groups=embed_dim), # pseudo attention
            nn.Conv2d(embed_dim, embed_dim, 1), # pseudo out linear
        ) if local_dw_ks > 0 else nn.Identity()
        
        # per-location embedding
        self.norm2 = norm_layer(embed_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, int(mlp_ratio*embed_dim), kernel_size=1),
            ResDWConvNCHW(int(mlp_ratio*embed_dim),ks=3) if mlp_dw else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(int(mlp_ratio*embed_dim), embed_dim, kernel_size=1)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # layer scale
        self.ls1 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nchw') \
            if layerscale > 0 else nn.Identity()
        self.ls2 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nchw') \
            if layerscale > 0 else nn.Identity()
        self.enlarger = nn.Sequential(
            nn.BatchNorm2d(embed_dim),
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim,embed_dim,kernel_size=3,padding=1)
        )
        # NOTE: hack, for visualization with forward hook
        self.vis_proxy = nn.Identity()
        
    def _forward_relation(self, x:torch.Tensor, init_slots:torch.Tensor):
        """
        x: (bs, c, h, w) tensor
        init_slots: (bs, num_slots, c) tensor
        """
        x_flatten = x.permute(0, 2, 3, 1).flatten(1, 2)
        logits = F.normalize(init_slots, p=2, dim=-1) @ \
            (self.slot_scale*F.normalize(x_flatten, p=2, dim=-1).transpose(-1, -2)) # (bs, num_slots, l)
        # soft grouping
        slots = torch.softmax(logits, dim=-1) @ x_flatten # (bs, num_slots, c)
        # cluster update with mha
        slots, attn_weights = self.relation_mha(query=slots, key=slots, value=slots, need_weights=False)

        if not self.training: # hack, for visualization
            logits, attn_weights = self.vis_proxy((logits, attn_weights))

        # soft ungrouping
        out = torch.softmax(logits.transpose(-1, -2), dim=-1) @ slots # (bs, h*w, c)
    
        out = out.permute(0, 2, 1).reshape_as(x) # (b, c, h, w)
        # # fuse with locally enhanced features
        out = out + self.feature_conv(x)

        return out, slots

    def forward(self, x:torch.Tensor):
        """
        x: (bs, c, h, w) tensor
        init_slots: (bs, num_slots, c) tensor
        """
        copy=x
        if self.slot_init == 'ada_avgpool':
            init_slots = F.adaptive_avg_pool2d(x, output_size=self.pool_size
                ).permute(0, 2, 3, 1).flatten(1, 2)
        elif self.slot_init == 'ada_maxpool':
            init_slots = F.adaptive_max_pool2d(x, output_size=self.pool_size
                ).permute(0, 2, 3, 1).flatten(1, 2)
        else:
            init_slots = self.init_slots

        # print(x.dtype, init_slots.dtype) -> float16 float32
        if self.with_conv_pos_emb:
            x = x + self.pos_conv(x)

        shortcut = x
        x, updt_slots = self._forward_relation(self.norm1(x), init_slots)
        x = shortcut + self.drop_path(self.ls1(x))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        return x,self.enlarger(copy-x)
    
    def extra_repr(self) -> str:
        return f"scale_mode={self.scale_mode}, "\
               f"slot_init={self.slot_init}, "\
               f"use_slot_attention={self.use_slot_attention}"

class myGLMix(nn.Module):
    """
    multihead attention with soft grouping + MLP
    """
    def __init__(self, 
        dim, num_heads, depth:int,
        mlp_ratio=4., drop_path=0.,
        ################
        mixing_mode='glmix', # {'mha',  'sgmha', 'dw', 'glmix'}
        local_dw_ks=5, # kernel size of dw conv, for 'dw' and 'glmix'
        slot_init:str='ada_avgpool', # {'param', 'conv', 'pool', 'ada_avgpool'}
        num_slots:int=64, # to control number of slots
        use_slot_attention:bool=True,
        cpe_ks:int=0,
        mlp_dw:bool=False,
        ##############
        layerscale=-1
        ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mixing_mode = mixing_mode
        self.blocks = nn.ModuleList([
                        myGLMixBlock(
                            embed_dim=dim,
                            num_heads=num_heads,
                            num_slots=num_slots,
                            slot_init=slot_init,
                            local_dw_ks=local_dw_ks,
                            use_slot_attention=use_slot_attention,
                            mlp_ratio=mlp_ratio,
                            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                            cpe_ks=cpe_ks,
                            mlp_dw=mlp_dw,
                            layerscale=layerscale
                        )for i in range(depth)])
        self.local=nn.ModuleList([(
                        DEBlock(dim)
                        )for i in range(depth)])
        self.merger = nn.Sequential(
            nn.Conv2d(dim*5,dim,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim,dim,kernel_size=1),
            LayerNorm2d(dim)
        )
        self.merger2 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
            batch_first=True)
        self.merger3 = nn.Sequential(
            nn.Conv2d(dim,dim*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim*2,dim,kernel_size=1),
        )
    def forward(self, x:torch.Tensor):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        m1,x1 = self.blocks[0](x)
        m1,x2 = self.blocks[1](m1)
        m1,x3 = self.blocks[2](m1)
        x0,x4 = self.blocks[3](m1)
        x1=self.local[0](x1)+x1
        x2=self.local[1](x2)+x2
        x3=self.local[2](x3)+x3
        x4=self.local[3](x4)+x4
        m1=self.merger(torch.cat([x0,x1,x2,x3,x4],dim=1)).permute(0, 2, 3, 1).flatten(1, 2)
        m2,_=self.merger2(query=m1, key=m1, value=m1, need_weights=False)
        #ipdb.set_trace()
        m2=m2.permute(0, 2, 1).reshape_as(x)
        return x0+self.merger3(m2)

class MHSA_NCHW_Block(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, dropout=0.,
                 mlp_ratio:float=4., drop_path:float=0., norm_layer=LayerNorm2d,
                 mlp_dw:bool=False, cpe_ks:int=0, # if > 0, the kernel size of the conv pos embedding
                 layerscale=-1,
                ) -> None:
        super().__init__()
        # convolutional position encoding
        self.with_conv_pos_emb = (cpe_ks > 0)
        if self.with_conv_pos_emb:
            self.pos_conv = nn.Conv2d(
                embed_dim, embed_dim,
                kernel_size=cpe_ks,
                padding=cpe_ks//2, groups=embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mha_op = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, int(mlp_ratio*embed_dim), kernel_size=1),
            ResDWConvNCHW(int(mlp_ratio*embed_dim),ks=3) if mlp_dw else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(int(mlp_ratio*embed_dim), embed_dim, kernel_size=1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ls1 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nlc') if layerscale > 0 else nn.Identity()
        self.ls2 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nchw') if layerscale > 0 else nn.Identity()

    def forward(self, x:torch.Tensor):
        """
        args:
            x: (bs, c, h, w) Tensor
        return:
            (bs, c, h, w) Tensor
        """
        # Conv. Pos. Embedding
        if self.with_conv_pos_emb:
            x = x + self.pos_conv(x)

        # reshape tp nlc format
        nchw_shape = x.size()
        x = x.permute(0, 2, 3, 1).flatten(1, 2) # (bs, h*w, c)

        # forward attention block
        # print(x.dtype)
        shortcut = x
        x = self.norm1(x)
        # print(x.dtype) -> float32 -> RuntimeError: expected scalar type Half but found Float
        # FIXME: below is just a workaround
        # https://github.com/NVIDIA/apex/issues/121#issuecomment-1235109690
        if not self.training:
            x = x.to(shortcut.dtype)
        x, attn_weights = self.mha_op(query=x, key=x, value=x, need_weights=False)
        x = shortcut + self.drop_path(self.ls1(x))
        
        # forward mlp block
        x = x.permute(0, 2, 1).reshape(nchw_shape)
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        return x


class BasicLayer(nn.Module):
    """
    Stack several Blocks (a stage of transformer)
    """
    def __init__(self, 
        dim, num_heads, depth:int,
        mlp_ratio=4., drop_path=0.,
        ################
        mixing_mode='glmix', # {'mha',  'sgmha', 'dw', 'glmix'}
        local_dw_ks=5, # kernel size of dw conv, for 'dw' and 'glmix'
        slot_init:str='ada_avgpool', # {'param', 'conv', 'pool', 'ada_avgpool'}
        num_slots:int=64, # to control number of slots
        use_slot_attention:bool=True,
        cpe_ks:int=0,
        mlp_dw:bool=False,
        ##############
        layerscale=-1
        ):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mixing_mode = mixing_mode

        # instantiate blocks
        if self.mixing_mode == 'mha':
            self.blocks = nn.ModuleList([
                MHSA_Block(
                    embed_dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layerscale=layerscale
                ) for i in range(depth)])
        elif self.mixing_mode == 'mha_nchw':
            self.blocks = nn.ModuleList([
                MHSA_NCHW_Block(
                    embed_dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    cpe_ks=cpe_ks,
                    mlp_dw=mlp_dw,
                    layerscale=layerscale
                ) for i in range(depth)])
        elif self.mixing_mode == 'glmix':
            self.blocks = nn.ModuleList([
                GLMixBlock(
                    embed_dim=dim,
                    num_heads=num_heads,
                    num_slots=num_slots,
                    slot_init=slot_init,
                    local_dw_ks=local_dw_ks,
                    use_slot_attention=use_slot_attention,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    cpe_ks=cpe_ks,
                    mlp_dw=mlp_dw,
                    layerscale=layerscale
                )for i in range(depth)])
        elif self.mixing_mode == 'glmix.mha_nchw': # hybrid
            self.blocks = nn.ModuleList([
                GLMixBlock(
                    embed_dim=dim,
                    num_heads=num_heads,
                    num_slots=num_slots,
                    slot_init=slot_init,
                    local_dw_ks=local_dw_ks,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    cpe_ks=cpe_ks,
                    mlp_dw=mlp_dw,
                    layerscale=layerscale
                ) if i % 2 == 0 else \
                MHSA_NCHW_Block(
                    embed_dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    cpe_ks=cpe_ks,
                    mlp_dw=mlp_dw,
                    layerscale=layerscale
                ) for i in range(depth)])
        else:
            raise ValueError('unknown block type')

    def forward(self, x:torch.Tensor):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        # print(x.dtype)
        if self.mixing_mode == 'mha':
            nchw_shape = x.size()
            x = x.permute(0, 2, 3, 1).flatten(1, 2)
            for blk in self.blocks:
                x = blk(x) # (bs, len, c)
            x = x.transpose(1, 2).reshape(nchw_shape)
        else: # the input output are both nchw format
            for blk in self.blocks:
                x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"



class NonOverlappedPatchEmbeddings(nn.ModuleList):
    def __init__(self, embed_dims:Iterable[int], in_chans=3,
                       midd_order='norm.proj',
                       norm_layer=nn.BatchNorm2d) -> None:
        assert midd_order in {'norm.proj', 'proj.norm'}
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims[0],kernel_size=(4, 4), stride=(4, 4)),
            norm_layer(embed_dims[0])
        )
        modules = [stem]
        for i in range(3):
            if midd_order == 'norm.proj':
                transition = nn.Sequential(
                    norm_layer(embed_dims[i]), 
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=(2, 2), stride=(2, 2)),
                )
            else:
                transition = nn.Sequential(
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=(2, 2), stride=(2, 2)),
                    norm_layer(embed_dims[i+1])
                )
            modules.append(transition)
        super().__init__(modules)



class OverlappedPacthEmbeddings(nn.ModuleList):
    def __init__(self, embed_dims:Iterable[int],
        in_chans=3,
        deep_stem=True,
        dual_patch_norm=False,
        midd_order='proj.norm',
        norm_layer=nn.BatchNorm2d
        ) -> None:
        assert midd_order in {'norm.proj', 'proj.norm'}
        if deep_stem:
            stem = nn.Sequential(
                LayerNorm2d(in_chans) if dual_patch_norm else nn.Identity(),
                nn.Conv2d(in_chans, embed_dims[0] // 2, kernel_size=3, stride=2, padding=1),
                norm_layer(embed_dims[0] // 2),
                nn.GELU(),
                nn.Conv2d(embed_dims[0] // 2, embed_dims[0], kernel_size=3, stride=2, padding=1),
                norm_layer(embed_dims[0])
            )
        else:
            stem = nn.Sequential(
                LayerNorm2d(in_chans) if dual_patch_norm else nn.Identity(),
                nn.Conv2d(in_chans, embed_dims[0] // 2, kernel_size=7, stride=4, padding=3),
                norm_layer(embed_dims[0] // 2),
            )
        modules = [stem]
        for i in range(3):
            if midd_order == 'norm.proj':
                transition = nn.Sequential(
                    norm_layer(embed_dims[i]), 
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=3, stride=2, padding=1),
                )
            else:
                transition = nn.Sequential(
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=3, stride=2, padding=1),
                    norm_layer(embed_dims[i+1]),
                )
            modules.append(transition)
        super().__init__(modules)

class ContinusParalleConv(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, pre_Batch_Norm=True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Conv_forward = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 1, padding=0, stride=1),
            nn.LeakyReLU())

    def forward(self, x):
        x = self.Conv_forward(x)
        return x
class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out
class YUVtoRGB(nn.Module):
    def __init__(self, standard="BT.601"):
        super(YUVtoRGB, self).__init__()
        
        # 根据标准初始化转换系数
        if standard == "BT.601":
            self.a = torch.tensor(0.299, dtype=torch.float32)
            self.b = torch.tensor(0.587, dtype=torch.float32)
            self.c = torch.tensor(0.114, dtype=torch.float32)
            self.d = torch.tensor(1.772, dtype=torch.float32)
            self.e = torch.tensor(1.402, dtype=torch.float32)
        elif standard == "BT.709":
            self.a = nn.Parameter(torch.tensor(0.2126, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(0.7152, dtype=torch.float32))
            self.c = nn.Parameter(torch.tensor(0.0722, dtype=torch.float32))
            self.d = nn.Parameter(torch.tensor(1.8556, dtype=torch.float32))
            self.e = nn.Parameter(torch.tensor(1.5748, dtype=torch.float32))
        elif standard == "BT.2020":
            self.a = nn.Parameter(torch.tensor(0.2627, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(0.6780, dtype=torch.float32))
            self.c = nn.Parameter(torch.tensor(0.0593, dtype=torch.float32))
            self.d = nn.Parameter(torch.tensor(1.8814, dtype=torch.float32))
            self.e = nn.Parameter(torch.tensor(1.4746, dtype=torch.float32))
        else:
            raise ValueError("Unsupported standard. Choose from 'BT.601', 'BT.709', or 'BT.2020'.")
        
    def forward(self, yuv):
        # 将 YUV 转换为 RGB
        Y, Cb, Cr = yuv[:, 0, :, :], yuv[:, 1, :, :], yuv[:, 2, :, :]
        
        R = Y + self.e * (Cr - 0.5)
        G = Y - (self.a * self.e / self.b) * (Cr - 0.5) - (self.c * self.d / self.b) * (Cb - 0.5)
        B = Y + self.d * (Cb - 0.5)
        
        # 将结果组合为 RGB 图像
        rgb = torch.stack((R, G, B), dim=1)
        return rgb
class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B,C,H, W = x.shape
        x=x.permute(0,2,3,1)
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x.permute(0,3,1,2)
class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # 96
        self.dim_scale = dim_scale  # 4
        #        96             384
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        #                          24
        self.norm = nn.LayerNorm(self.dim // dim_scale)
        self.conv= nn.Conv2d(self.dim//4, 3, 1)

    def forward(self, x):

        B, C,H, W,= x.shape
        x=x.permute(0,2,3,1)
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)
        x=self.conv(x.permute(0,3,1,2))
        return x
class Final_pixel_shuffle2D(nn.Module):
    def __init__(self, dim,norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # 96
        self.expand = nn.Linear(self.dim, 4 * self.dim, bias=False)
        self.norm = nn.LayerNorm(4 * self.dim)
        self.conv= nn.Conv2d(self.dim, 3, 1)
    def forward(self, x):
        #ipdb.set_trace()
        x=x.permute(0,2,3,1)
        x = self.expand(x)
        x = self.norm(x)
        x=x.permute(0,3,1,2)
        x=F.pixel_shuffle(x, upscale_factor=2)
        x=self.conv(x)
        return x
class Final_pixel_shuffle2DYCBCR(nn.Module):
    def __init__(self, dim,norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # 96
        self.expandy = nn.Linear(self.dim//2, 2 * self.dim, bias=False)
        self.expandc = nn.Linear(self.dim//2, 2 * self.dim, bias=False)
        self.normc = nn.LayerNorm(2 * self.dim)
        self.normy = nn.LayerNorm(2 * self.dim)
        self.convc= nn.Conv2d(self.dim//2, 2, 1)
        self.convy= nn.Conv2d(self.dim//2, 1, 1)
        self._ycbcr_to_rgb=YUVtoRGB()
    def forward(self, x):
        #ipdb.set_trace()
        x=x.permute(0,2,3,1)
        y,c=x.chunk(2,dim=-1)
        c = self.normc(self.expandc(c))
        y = self.normy(self.expandy(y))
        c=c.permute(0,3,1,2)
        y=y.permute(0,3,1,2)
        y=F.pixel_shuffle(y, upscale_factor=2)
        c=F.pixel_shuffle(c, upscale_factor=2)
        c=self.convc(c)
        y=self.convy(y)
        return self._ycbcr_to_rgb(torch.cat([y,c],dim=1))
class myFinal_pixel_shuffle2Dycbcr(nn.Module):
    def __init__(self, dim,norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # 96
        self.expandy = nn.Linear(self.dim//2, 2 * self.dim, bias=False)
        self.expandc = nn.Linear(self.dim//2, 2 * self.dim, bias=False)
        self.normc = nn.LayerNorm(2 * self.dim)
        self.normy = nn.LayerNorm(2 * self.dim)
        #self.conv= myGLMix(dim,4,5,slot_init='ada_maxpool')
        self.conv=nn.Conv2d(dim,dim,1)
    def forward(self, x):
        x=x.permute(0,2,3,1)
        y,c=x.chunk(2,dim=-1)
        c = self.normc(self.expandc(c))
        y = self.normy(self.expandy(y))
        c=c.permute(0,3,1,2)
        y=y.permute(0,3,1,2)
        y=F.pixel_shuffle(y, upscale_factor=2)
        c=F.pixel_shuffle(c, upscale_factor=2)
        x=torch.cat([y,c],dim=1)
        x=self.conv(x)
        #ipdb.set_trace()
        return x

class myFinal_pixel_shuffle2D(nn.Module):
    def __init__(self, dim,norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # 96
        self.expand = nn.Linear(self.dim, 4 * self.dim, bias=False)
        self.norm = nn.LayerNorm(4 * self.dim)
        self.conv= myGLMix(dim,4,5,slot_init='ada_maxpool')
    def forward(self, x):
        x=x.permute(0,2,3,1)
        x = self.expand(x)
        x = self.norm(x)
        x=x.permute(0,3,1,2)
        x=F.pixel_shuffle(x, upscale_factor=2)
        x=self.conv(x)
        #ipdb.set_trace()
        return x
class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
class GLNet(nn.Module):
    """
    vision transformer with soft grouping
    """
    def __init__(self,
        in_chans=3,
        num_classes=3,
        depth=[2, 2, 6, 2],
        embed_dim=[96, 192, 384, 768],
        head_dim=32, qk_scale=None,
        drop_path_rate=0., drop_rate=0.,
        use_checkpoint_stages=[],
        mlp_ratios=[4, 4, 4, 4],
        norm_layer=LayerNorm2d,
        pre_head_norm_layer=None,
        ######## glnet specific ############
        mixing_modes=('glmix', 'glmix', 'glmix', 'mha'), # {'mha', 'glmix',  'glmix.mha_nchw', 'mha_nchw'}
        local_dw_ks=5, # kernel size of dw conv
        slot_init:str='param', #{'param', 'conv', 'pool', 'ada_pool'}
        num_slots:int=64, # to control number of slots
        cpe_ks:int=0,
        #######################################
        downsample_style:str='non_ovlp', # {'non_ovlp', 'ovlp'}
        transition_layout:str='proj.norm', # {'norm.proj', 'proj.norm'}
        dual_patch_norm:bool=False,
        mlp_dw:bool=False,
        layerscale:float=-1.,
        ###################
        **unused_kwargs
        ):
        super().__init__()
        print(f"unused_kwargs in model initilization: {unused_kwargs}.")

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ############ downsample layers (patch embeddings) ######################
        assert downsample_style in {'non_ovlp', 'ovlp'}
        if downsample_style=='ovlp':
            self.downsample_layers = OverlappedPacthEmbeddings(
                embed_dims=embed_dim, in_chans=in_chans, norm_layer=norm_layer,
                midd_order=transition_layout,
                dual_patch_norm=dual_patch_norm)
        else:
            self.downsample_layers = NonOverlappedPatchEmbeddings(
                embed_dims=embed_dim, in_chans=in_chans, norm_layer=norm_layer,
                midd_order=transition_layout
            )
        ##########################################################################
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.up = nn.ModuleList()
        nheads= [dim // head_dim for dim in embed_dim]
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        local_dw_ks = [local_dw_ks,]*4 if isinstance(local_dw_ks, int) else local_dw_ks
        self.conv_uplayer = ResidualUpSample(embed_dim[1])
        self.downsample_convlayer = ContinusParalleConv(embed_dim[0],embed_dim[1], pre_Batch_Norm=True)
        for i in range(2):
            if i ==0:
                stage=myGLMix(embed_dim[i],nheads[i],4,slot_init='ada_maxpool')
            else:
                stage = BasicLayer(
                dim=embed_dim[i],
                depth=depth[i],
                num_heads=nheads[i], 
                mlp_ratio=mlp_ratios[i],
                drop_path=dp_rates[sum(depth[:i]):sum(depth[:i+1])],
                ####### glnet specific ########
                mixing_mode=mixing_modes[i],
                local_dw_ks=local_dw_ks[i],
                slot_init='ada_maxpool',
                num_slots=num_slots,
                cpe_ks=cpe_ks,
                mlp_dw=mlp_dw,
                layerscale=layerscale
                ##################################
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
        for i in range(2,3):
            upsample=PatchExpand2D(embed_dim[i])
            self.up.append(upsample)
        for i in range(2,4):
            if i ==3:
                stage=myGLMix(embed_dim[i],nheads[i],4,slot_init='ada_avgpool')
            else:
                stage = BasicLayer(
                dim=embed_dim[i],
                depth=depth[i],
                num_heads=nheads[i], 
                mlp_ratio=mlp_ratios[i],
                drop_path=dp_rates[sum(depth[:i]):sum(depth[:i+1])],
                ####### glnet specific ########
                mixing_mode=mixing_modes[i],
                local_dw_ks=local_dw_ks[i],
                slot_init='ada_avgpool',
                num_slots=num_slots,
                cpe_ks=cpe_ks,
                mlp_dw=mlp_dw,
                layerscale=layerscale
                ##################################
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
        ##########################################################################
        pre_head_norm = pre_head_norm_layer or norm_layer 
        self.norm = pre_head_norm(embed_dim[-1])
        # Classifier head
        self.pool = nn.MaxPool2d(2)
        self.final_up = Final_pixel_shuffle2DYCBCR(dim=embed_dim[-1], norm_layer=norm_layer)
        self.final_up_before = myFinal_pixel_shuffle2Dycbcr(dim=embed_dim[-1], norm_layer=norm_layer)
        self.resconv=nn.Conv2d(3,3,1)
        self.scale=nn.Parameter(torch.ones(1))
        self.apply(self._init_weights)
    def _rgb_to_ycbcr(self, image):
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5
        
        yuv = torch.stack((y, u, v), dim=1)
        return yuv
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        skip_list = []
        for i in range(2):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            skip_list.append(x)
        #import ipdb;ipdb.set_trace()
        return x,skip_list

    def downsample_conv(self,x):
        x = self.downsample_convlayer(x)
        x = self.pool(x)
        return x
    def uplayer_conv(self,x):
        x = self.conv_uplayer(x)
        return x

    def forward_features_up(self, x, skip_list):
        #import ipdb;ipdb.set_trace()
        for i in range(2,4):
            if i == 2:
                x = self.stages[i](x+self.downsample_conv(skip_list[0]))
                x = self.up[0](x)
            elif i == 3:
                x = self.stages[i](x + skip_list[0]) 
                x = x+ self.uplayer_conv(skip_list[1])
        return x
    
    def forward_final(self, x,y):
        # input 3*64*64*96   out=3 256 256 24
        x=self.final_up_before(x)
        x = self.final_up(x)+self.scale*y
        x=self.resconv(x)
        # out=3 24 256 256
        return x

    def forward(self, y:torch.Tensor):
        ycbcr=self._rgb_to_ycbcr(y)
        x, skip_list = self.forward_features(ycbcr)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x,y)
        return x




@register_model
def glnet_4g(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = GLNet(
        depth=[4, 8, 8, 4],
        embed_dim=[96, 192, 192, 96],
        mlp_ratios=[4,4,4,4],
        head_dim=16,
        norm_layer=nn.BatchNorm2d,
        ######## glnet specific ############
        mixing_modes=('glmix', 'glmix.mha_nchw', 'glmix.mha_nchw', 'glmix'),
        local_dw_ks=5, # kernel size of dw conv
        slot_init='param', #{'param', 'conv', 'pool', 'ada_maxpool','ada_avgpool'}
        num_slots=64, # to control number of slots
        #######################################
        cpe_ks=3,
        downsample_style='ovlp', # overlapped patch embedding
        transition_layout='proj.norm',
        mlp_dw=True,
        #######################################
        **kwargs)
    return model


class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.VUnet = glnet_4g()
    def forward(self, inputs):
        out =self.VUnet(inputs)
        return out