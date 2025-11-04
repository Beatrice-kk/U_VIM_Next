from networks.efficientunet import Effi_UNet
from networks.enet import ENet
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_DS, UNet_URPC, UNet_CCT
import argparse
import torch.nn as nn
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.config import get_config
from networks.nnunet import initialize_network

from networks.vision_mamba import MambaUnet
from models.dynamic_mamba_mask import DynamicMambaMaskedModel

import yaml
from types import SimpleNamespace


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Cross_Supervision_CNN_Trans2D', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--if_semi', action='store_true',
                    help='if semi-supervised')

# 延迟解析参数，避免在导入时执行
# args = parser.parse_args()
# config = get_config(args)


def net_factory(net_type="unet", in_chns=1, class_num=4):
   if net_type == "unet":
      net = UNet(in_chns=in_chns, class_num=class_num).cuda()
   elif net_type == "enet":
      net = ENet(in_channels=in_chns, num_classes=class_num).cuda()
   elif net_type == "unet_ds":
      net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
   elif net_type == "unet_cct":
      net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
   elif net_type == "unet_urpc":
      net = UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()
   elif net_type == "efficient_unet":
      net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                     in_channels=in_chns, classes=class_num).cuda()
   elif net_type == "ViT_Seg":
      # 如果需要在导入时使用，需要解析参数
      # 这里暂时使用默认配置，或者从调用者传入
      from config import get_config as get_config_main
      temp_args = argparse.Namespace(
         cfg="../code/configs/vmamba_tiny.yaml",
         opts=None,
         zip=False,
         cache_mode='part',
         resume='',
         accumulation_steps=0,
         use_checkpoint=False,
         amp_opt_level='',
         tag='',
         eval=False,
         throughput=False,
         batch_size=8
      )
      config = get_config_main(temp_args)
      net = ViT_seg(config, img_size=[224, 224],  # 使用默认patch_size
                     num_classes=class_num).cuda()
   elif net_type == "pnet":
      net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
   elif net_type == "nnUNet":
      net = initialize_network(num_classes=class_num).cuda()

   # elif net_type == "mambaunet":
   #    # net = MambaUnet(in_chns=in_chns, class_num=class_num).cuda()
   #    from configs.swin_tiny_patch4_window7_224_lite import get_config  # <--- 加导入
   #    config = get_config()  # <--- 加载 config
   #    net = MambaUnet(config=config, num_classes=class_num, img_size=256).cuda()
   #    print("使用 MambaUnet 作为网络结构")
   #    print("使用 MambaUnet 作为网络结构")
   #    print("使用 MambaUnet 作为网络结构")
   #    print("使用 MambaUnet 作为网络结构")
   #    print("使用 MambaUnet 作为网络结构")
      
         
   elif net_type == "mambaunet":

      
      yaml_path = "../code/configs/vmamba_tiny.yaml"
      with open(yaml_path, 'r') as f:
         config_dict = yaml.safe_load(f)
      
      config_dict['MODEL']['VSSM']['IN_CHANS'] = 3
      config_dict['MODEL']['VSSM']['NUM_CLASSES'] = class_num
      
      def to_obj(d):
         return SimpleNamespace(**{k: to_obj(v) for k, v in d.items()}) if isinstance(d, dict) else d
      
      config = to_obj(config_dict)
      
      net = MambaUnet(config=config, img_size=256).cuda()
   elif net_type == "dynamic_mask_mamba" or net_type == "mambaunet_dynamicmask":
      # 创建动态掩码 Mamba 模型
      yaml_path = "../code/configs/vmamba_tiny.yaml"
      with open(yaml_path, 'r') as f:
         config_dict = yaml.safe_load(f)
      
      config_dict['MODEL']['VSSM']['IN_CHANS'] = 3
      config_dict['MODEL']['VSSM']['NUM_CLASSES'] = class_num
      
      def to_obj(d):
         return SimpleNamespace(**{k: to_obj(v) for k, v in d.items()}) if isinstance(d, dict) else d
      
      config = to_obj(config_dict)
      
      # 创建基础 MambaUnet
      base_model = MambaUnet(config=config, img_size=256).cuda()
      # 尝试加载预训练权重（基础模型）
      try:
         from config import get_config as get_config_main
         temp_args = argparse.Namespace(
            cfg=yaml_path,
            opts=None,
            zip=False,
            cache_mode='part',
            resume='',
            accumulation_steps=0,
            use_checkpoint=False,
            amp_opt_level='',
            tag='',
            eval=False,
            throughput=False,
            batch_size=8
         )
         config_full = get_config_main(temp_args)
         base_model.load_from(config_full)
      except Exception as e:
         print(f"Warning: Could not load pretrained weights for base model: {e}")
      
      # 用 DynamicMambaMaskedModel 包装
      masked_model = DynamicMambaMaskedModel(base_model, in_channels=in_chns, epsilon=0.1).cuda()
      
      # 创建一个包装器，使测试时只返回 logits（兼容 test_single_volume）
      class TestWrapper(nn.Module):
         def __init__(self, masked_model):
            super().__init__()
            self.masked_model = masked_model
         
         def forward(self, x):
            logits, _ = self.masked_model(x)
            return logits
      
      net = TestWrapper(masked_model).cuda()
   else:
      net = None
   return net
