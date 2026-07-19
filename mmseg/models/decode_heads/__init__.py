# Copyright (c) OpenMMLab. All rights reserved.
from .ann_head import ANNHead
from .apc_head import APCHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .ddr_head import DDRHead
from .dm_head import DMHead
from .dnl_head import DNLHead
from .dpt_head import DPTHead
from .ema_head import EMAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .ppp_head import PPPHead
from .ppp_head_v2 import PPPHead_V2
from .fpn_head import FPNHead
from .gc_head import GCHead
from .ham_head import LightHamHead
from .isa_head import ISAHead
from .knet_head import IterativeDecodeHead, KernelUpdateHead, KernelUpdator
from .lraspp_head import LRASPPHead
from .mask2former_head import Mask2FormerHead
from .maskformer_head import MaskFormerHead
from .nl_head import NLHead
from .ocr_head import OCRHead
from .pid_head import PIDHead
from .point_head import PointHead
from .psa_head import PSAHead
from .psp_head import PSPHead
from .san_head import SideAdapterCLIPHead
from .segformer_head import SegformerHead
from .segmenter_mask_head import SegmenterMaskTransformerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .setr_mla_head import SETRMLAHead
from .setr_up_head import SETRUPHead
from .stdc_head import STDCHead
from .uper_head import UPerHead
from .vpd_depth_head import VPDDepthHead
from .ppp_mark_head import PPPMarkHead
from .ppp_mark_uper_head import PPPMarkUperHead
from .ppp_mark_head_wo_bottlenck import PPPMarkHeadWoBottleneck
from .ppp_mark_unet_head import PPPMarkUNetHead
from .ppp_mark_vit_upsample_head import PPPMarkViTUpsampleHead
from .ppp_mark_vit_patch_rearrange_head import PPPMarkViTPatchRearrangeHead
from .ppp_mark_vit_patch_rearrange_multitask_head import PPPMarkViTPatchRearrangeMultiTaskHead
from .ppp_mark_vit_patch_rearrange_multitask_twodepth_head import PPPMarkViTPatchRearrangeMultiTaskTwoDepthHead
from .ppp_mark_vit_patch_rearrange_head_disparity_depth import PPPMarkViTPatchRearrangeDisparityDepthHead
from .ppp_mark_unet_depth_loss_head import PPPMarkUNetDepthLossHead
from .ppp_polar_liif_ppp_head import PolarLIIFPPPHead
from .ppp_finals_polar_liif_head import PPPfinalsPolarLIIFPPPHead
from .ppp_finals_unet_head import PPPfinalsUNetDepthLossHead
from .ppp_finals_uper_head import PPPfinalsUperHead
from .ppp_finals_vit_patch_rearrange_head_polarBEV import PPPPolarViTPatchRearrangeHead
from .ppp_finals_vit_patch_rearrange_head import PPPfinalsViTPatchRearrangeHead

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
    'PointHead', 'APCHead', 'DMHead', 'LRASPPHead', 'SETRUPHead',
    'SETRMLAHead', 'DPTHead', 'SETRMLAHead', 'SegmenterMaskTransformerHead',
    'SegformerHead', 'ISAHead', 'STDCHead', 'IterativeDecodeHead',
    'KernelUpdateHead', 'KernelUpdator', 'MaskFormerHead', 'Mask2FormerHead',
    'LightHamHead', 'PIDHead', 'DDRHead', 'VPDDepthHead', 'SideAdapterCLIPHead',
    'PPPHead', 'PPPHead_V2', 'PPPMarkHead', 'PPPMarkUperHead', 'PPPMarkHeadWoBottleneck',
    'PPPMarkUNetHead', 'PPPMarkViTUpsampleHead', 'PPPMarkViTPatchRearrangeHead',
    'PPPMarkViTPatchRearrangeMultiTaskHead', 'PPPMarkViTPatchRearrangeMultiTaskTwoDepthHead',
    'PPPMarkViTPatchRearrangeDisparityDepthHead', 'PPPMarkUNetDepthLossHead',
    'PolarLIIFPPPHead', 'PPPfinalsPolarLIIFPPPHead', 'PPPfinalsUperHead', 'PPPPolarViTPatchRearrangeHead',
    'PPPfinalsViTPatchRearrangeHead', 'PPPfinalsUNetDepthLossHead'
]
