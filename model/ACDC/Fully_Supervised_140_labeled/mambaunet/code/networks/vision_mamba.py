# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .mamba_sys import VSSM, SS2D
from einops import rearrange

logger = logging.getLogger(__name__)


class ParallelConvMambaBlock(nn.Module):
    """并行卷积层和Mamba交互模块"""
    def __init__(self, dim, d_state=16, kernel_size=3):
        super(ParallelConvMambaBlock, self).__init__()
        self.dim = dim
        
        # 并行卷积层分支
        self.conv_branch = nn.Sequential(
            Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim),
            BatchNorm2d(dim),
            ReLU(inplace=True),
            Conv2d(dim, dim, kernel_size=1),
            BatchNorm2d(dim)
        )
        
        # Mamba交互分支
        self.mamba_branch = SS2D(d_model=dim, d_state=d_state, dropout=0.0)
        self.norm = LayerNorm(dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            Linear(dim * 2, dim),
            LayerNorm(dim)
        )
        
    def forward(self, x):
        # x shape: B, H, W, C
        B, H, W, C = x.shape
        
        # 转换为卷积格式: B, C, H, W
        x_conv = rearrange(x, 'b h w c -> b c h w')
        conv_out = self.conv_branch(x_conv)
        conv_out = rearrange(conv_out, 'b c h w -> b h w c')
        
        # Mamba分支处理
        mamba_out = self.mamba_branch(self.norm(x))
        
        # 融合两个分支
        fused = torch.cat([conv_out, mamba_out], dim=-1)  # B, H, W, 2C
        fused = self.fusion(fused)  # B, H, W, C
        
        # 残差连接
        output = x + fused
        return output


class MambaUnet(nn.Module):
   # def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
   def __init__(self, config, img_size=224, num_classes=4, zero_head=False, vis=False):
   
        super(MambaUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.mamba_unet =  VSSM(
                              patch_size=config.MODEL.VSSM.PATCH_SIZE,
                              in_chans=config.MODEL.VSSM.IN_CHANS,
                              num_classes=self.num_classes,
                              embed_dim=config.MODEL.VSSM.EMBED_DIM,
                              depths=config.MODEL.VSSM.DEPTHS,
                           #   mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                              drop_rate=config.MODEL.DROP_RATE,
                              drop_path_rate=config.MODEL.DROP_PATH_RATE,
                              # patch_norm=config.MODEL.SWIN.PATCH_NORM,
                              use_checkpoint=config.TRAIN.USE_CHECKPOINT
                              )
        
        # 为解码器添加并行卷积和Mamba交互模块
        # 获取解码器各层的维度
        embed_dim = config.MODEL.VSSM.EMBED_DIM
        depths = config.MODEL.VSSM.DEPTHS
        num_layers = len(depths)
        
        # 为每个解码器层创建并行模块
        # 解码器各层在layer_up之后的输出维度：
        # inx=0: PatchExpand输出 embed_dim * 2 ** (num_layers - 2) = 384
        # inx=1: VSSLayer_up (有upsample) 输入384，输出192（PatchExpand将维度减半）
        # inx=2: VSSLayer_up (有upsample) 输入192，输出96（PatchExpand将维度减半）
        # inx=3: VSSLayer_up (无upsample) 输入96，输出96
        self.parallel_blocks = nn.ModuleList()
        for i_layer in range(num_layers):
            if i_layer == 0:
                # 第一个层是PatchExpand，输出维度是 embed_dim * 2 ** (num_layers - 2)
                layer_dim = int(embed_dim * 2 ** (num_layers - 2))
            elif i_layer < num_layers - 1:
                # 中间层有upsample，输出维度是输入维度的一半
                # 输入维度是 embed_dim * 2 ** (num_layers - 1 - i_layer)
                # 输出维度是 embed_dim * 2 ** (num_layers - 2 - i_layer)
                layer_dim = int(embed_dim * 2 ** (num_layers - 2 - i_layer))
            else:
                # 最后一层没有upsample，输出维度等于输入维度
                layer_dim = int(embed_dim * 2 ** (num_layers - 1 - i_layer))
            
            # 为所有层创建并行块（包括第一个层）
            parallel_block = ParallelConvMambaBlock(dim=layer_dim, d_state=max(16, layer_dim // 6))
            self.parallel_blocks.append(parallel_block)

   def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        
        # 获取编码器特征
        x, x_downsample = self.mamba_unet.forward_features(x)
        x = self.mamba_unet.norm(x)  # B H W C
        
        # 解码器前向传播，添加并行处理
        for inx, layer_up in enumerate(self.mamba_unet.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3-inx]], -1)
                x = self.mamba_unet.concat_back_dim[inx](x)
                x = layer_up(x)
            
            # 在解码器层之后添加并行卷积和Mamba交互
            # 对所有解码器层应用并行块
            if inx < len(self.parallel_blocks):
                x = self.parallel_blocks[inx](x)
        
        x = self.mamba_unet.norm_up(x)  # B H W C
        
        # 最终上采样和输出
        logits = self.mamba_unet.up_x4(x)
        return logits

   def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.mamba_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.mamba_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.mamba_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")