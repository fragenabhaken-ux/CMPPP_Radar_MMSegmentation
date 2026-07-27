module --force purge
module load Stages/2025  GCCcore/.13.3.0
module load tensorboard

choice=${1:-0}

case "$choice" in
  0)
    tensorboard --logdir_spec="\
dinov3_LoRA_V3_polarLIIF_5k_inits:/e/scratch/nxtaim-1/huber7/work_dirs/952081_ppp_dinov3_LoRA_V3_polar_liif_init_test/20260623_022607/vis_data/"\
    --port=12355 serve
    ;;
  1)
    tensorboard --logdir_spec="\
unet:/p/scratch/nxtaim-1/huber7/work_dirs/13763261_unet_160k/20260425_004229/vis_data/,\
unet_8nodes:/p/scratch/nxtaim-1/huber7/work_dirs/13767623_unet_160k_8nodes/20260427_013913/vis_data/" \
    --port=12301 serve
    ;;
  2)
    tensorboard --logdir_spec="\
dinov3_LoRA_V3_log:/e/scratch/nxtaim-1/huber7/work_dirs/889927_ppp_dinov3_LoRA_V3_25k/20260615_231613/vis_data/,\
dinov3_LoRA_V3_disparity:/e/scratch/nxtaim-1/huber7/data/260617_dinov3_LoRA_V3/906347_ppp_dinov3_LoRA_V3_disparity_25k/20260617_043344/vis_data/,\
unet_50k:/e/scratch/nxtaim-1/huber7/work_dirs/858912_ppp_unet_50k/20260614_011350/vis_data/"\
    --port=12355 serve
    ;;
    3)
    tensorboard --logdir_spec="\
dinov3_LoRA_V3_fullMark_vel10:/e/scratch/nxtaim-1/huber7/work_dirs/923580_ppp_dinov3_LoRA_V3_fullMark_init_test_10/20260618_221959/vis_data/,\
dinov3_LoRA_V3_fullMark_vel100:/e/scratch/nxtaim-1/huber7/work_dirs/923568_ppp_dinov3_LoRA_V3_fullMark_init_test_100/20260618_221127/vis_data/,\
dinov3_LoRA_V3_fullMark_absVel10:/e/scratch/nxtaim-1/huber7/work_dirs/923623_ppp_dinov3_LoRA_V3_fullMarkAbsVel_init_test/20260618_225217/vis_data/"\
    --port=12355 serve
    ;;
    4)
    tensorboard --logdir_spec="\
dinov3_LoRA_V3_polarLIIF_5k_inits:/e/scratch/nxtaim-1/huber7/work_dirs/945615_ppp_dinov3_LoRA_V3_polarLIIF_5k_init/20260622_122331/vis_data/,\
dinov3_LoRA_V3_fullMark_absVel_LRRangeTest:/e/scratch/nxtaim-1/huber7/work_dirs/939979_ppp_dinov3_LoRA_V3_fullMarkAbsVel_5k_LRrangetest/20260621_210435/vis_data/"\
    --port=12355 serve
    ;;  
    5)
    tensorboard --logdir_spec="\
dinov3_LoRA_V3_fullMarkAbsVel_25k:/e/scratch/nxtaim-1/huber7/work_dirs/946592_ppp_dinov3_LoRA_V3_fullMarkAbsVel_25k/20260622_154135/"\
    --port=12355 serve
    ;;
    6)
    tensorboard --logdir_spec="\
UNet_50k:/e/scratch/nxtaim-1/huber7/data/260615_unet_50k/858912_ppp_unet_50k/20260614_011350/vis_data/,\
dinov3_polar_log_depth:/e/scratch/nxtaim-1/huber7/data/260617_dinov3_LoRA_V3/889927_ppp_dinov3_LoRA_V3_25k/20260615_231613/vis_data/,\
dinov3_polar_disp_depth:/e/scratch/nxtaim-1/huber7/data/260617_dinov3_LoRA_V3/906347_ppp_dinov3_LoRA_V3_disparity_25k/20260617_043344/vis_data/,\
dinov3_polar_log_depth_velAbs:/e/scratch/nxtaim-1/huber7/data/260617_dinov3_LoRA_V3/946592_ppp_dinov3_LoRA_V3_fullMarkAbsVel_25k/20260622_154135/vis_data/"\
    --port=12355 serve
    ;;  
    7)
    tensorboard --logdir_spec="\
upernet:/e/scratch/nxtaim-1/huber7/work_dirs/1009042_ppp_upernet_smoke/20260721_173031/vis_data/,\
unet:/e/scratch/nxtaim-1/huber7/work_dirs/1010204_ppp_unet_smoke/20260721_223835/vis_data/,\
TrafoEnc_CamDom:/e/scratch/nxtaim-1/huber7/work_dirs/1010579_ppp_TrafoEnc_BEVpolDom/20260722_005045/vis_data/,\
TrafoEnc_BEVpolDom:/e/scratch/nxtaim-1/huber7/work_dirs/1010580_ppp_TrafoEnc_BEVpolDom/20260722_005045/vis_data/,\
TrafoEncDec:/e/scratch/nxtaim-1/huber7/work_dirs/1010647_ppp_TrafoEncDec/20260722_024640/vis_data/"\
    --port=12355 serve
    ;;
    8)
    tensorboard --logdir_spec="\
UNet:/e/scratch/nxtaim-1/huber7/work_dirs/1041277_ppp_finals_unet_54k/20260725_062756/vis_data/,\
UNet_woVel:/e/scratch/nxtaim-1/huber7/work_dirs/1041278_ppp_finals_unet_54k_woVel/20260725_062804/vis_data/,\
UperNet:/e/scratch/nxtaim-1/huber7/work_dirs/1041089_ppp_finals_upernet_47k/20260725_060533/vis_data/,\
TrafoEnc_CamDom:/e/scratch/nxtaim-1/huber7/work_dirs/1041091_ppp_finals_TrafoEnc_CamDom_24k/20260725_060602/vis_data/,\
TrafoEnc_BEVpolDom:/e/scratch/nxtaim-1/huber7/work_dirs/1041092_ppp_finals_TrafoEnc_BEVpolDom_20k/20260725_060631/vis_data/,\
TrafoEncDec:/e/scratch/nxtaim-1/huber7/work_dirs/1041329_ppp_finals_TrafoEncDec_22k/20260725_072222/vis_data/,\
TrafoEncDec_woVel:/e/scratch/nxtaim-1/huber7/work_dirs/1041328_ppp_finals_TrafoEncDec_22k_woVel/20260725_072209/vis_data/"\
    --port=12355 serve
    ;;
  *)
    echo "Invalid option: $choice"
    echo "Usage: $0 [0-9]"
    exit 1
    ;;
esac