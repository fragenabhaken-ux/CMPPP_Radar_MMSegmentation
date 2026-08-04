module --force purge
module load Stages/2025  GCCcore/.13.3.0
module load tensorboard

choice=${1:-0}

case "$choice" in
  0)
    tensorboard --logdir_spec="\
TrafoEncDec:/e/scratch/nxtaim-1/huber7/work_dirs/1041329_ppp_finals_TrafoEncDec_22k/20260725_072222/vis_data/,\ 
TrafoEncDec_run2:/e/scratch/nxtaim-1/huber7/work_dirs/1065402_ppp_finals_TrafoEncDec_run2_20k/20260727_201415/vis_data/"\
    --port=12356 serve
    ;;
  1)
    tensorboard --logdir_spec="\
TrafoEncDec:/e/scratch/nxtaim-1/huber7/work_dirs/1041329_ppp_finals_TrafoEncDec_22k/20260725_072222/vis_data/" \
    --port=12301 serve
    ;;
  2)
    tensorboard --logdir_spec="\
UNet:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1041277_ppp_finals_unet_54k/20260725_062756/vis_data/,\
UNet_woVel:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1041278_ppp_finals_unet_54k_woVel/20260725_062804/vis_data/,\ 
UNet_woVel_woRCS:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1047407_ppp_finals_unet_54k_woVel_woRCS/20260726_145120/vis_data/,\
TrafoEnc_CamDom:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1041091_ppp_finals_TrafoEnc_CamDom_24k/20260725_060602/vis_data/,\ 
TrafoEnc_BEVpolDom:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1041092_ppp_finals_TrafoEnc_BEVpolDom_20k/20260725_060631/vis_data/,\ 
TrafoEncDec:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_2/1065402_ppp_finals_TrafoEncDec_run2_20k/20260727_201415/vis_data/,\ 
UperNet:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_2/1096822_ppp_finals_upernet_run3_50k/20260730_015303/vis_data/"\
    --port=12357 serve
    ;;
    3)
    tensorboard --logdir_spec="\
TrafoEnc_BEVpolDom:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1041092_ppp_finals_TrafoEnc_BEVpolDom_20k/20260725_060631/vis_data/,\ 
TrafoEncDec:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_2/1065402_ppp_finals_TrafoEncDec_run2_20k/20260727_201415/vis_data/"\
    --port=12355 serve
    ;;
    4)
    tensorboard --logdir_spec="\
UperNet_run1:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1041089_ppp_finals_upernet_47k/20260725_060533/vis_data/,\ 
UperNet_run2:/e/scratch/nxtaim-1/huber7/work_dirs/1069825_ppp_finals_upernet_run2_21k/20260728_022912/vis_data/"\
    --port=12355 serve
    ;;  
    5)
    tensorboard --logdir_spec="\
UperNet:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_2/1096822_ppp_finals_upernet_run3_50k/20260730_015303/vis_data/"\
    --port=12355 serve
    ;;
    6)
    tensorboard --logdir_spec="\
UNet:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1041277_ppp_finals_unet_54k/20260725_062756/vis_data/,\ 
UNet_woVel:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1041278_ppp_finals_unet_54k_woVel/20260725_062804/vis_data/,\ 
UNet_woVel_woRCS:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1047407_ppp_finals_unet_54k_woVel_woRCS/20260726_145120/vis_data/,\
UperNet:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_2/1096822_ppp_finals_upernet_run3_50k/20260730_015303/vis_data/,\ 
TrafoEnc_CamDom:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1041091_ppp_finals_TrafoEnc_CamDom_24k/20260725_060602/vis_data/,\ 
TrafoEnc_BEVpolDom:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1041092_ppp_finals_TrafoEnc_BEVpolDom_20k/20260725_060631/vis_data/,\ 
TrafoEncDec:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_2/1065402_ppp_finals_TrafoEncDec_run2_20k/20260727_201415/vis_data/,\ 
TrafoEncDec_woVel:/e/scratch/nxtaim-1/huber7/work_dirs/1096858_ppp_finals_TrafoEncDec_run2_20k_woVel/20260730_015622/vis_data/"\
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
UNet_OLD:/e/scratch/nxtaim-1/huber7/work_dirs/1041277_ppp_finals_unet_54k/20260725_062756/vis_data/,\ 
UNet_woVel:/e/scratch/nxtaim-1/huber7/work_dirs/1041278_ppp_finals_unet_54k_woVel/20260725_062804/vis_data/,\ 
UNet_woVel_woRCS:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1047407_ppp_finals_unet_54k_woVel_woRCS/20260726_145120/vis_data,\ 
UperNet:/e/scratch/nxtaim-1/huber7/work_dirs/1041089_ppp_finals_upernet_47k/20260725_060533/vis_data/,\ 
TrafoEnc_CamDom:/e/scratch/nxtaim-1/huber7/work_dirs/1041091_ppp_finals_TrafoEnc_CamDom_24k/20260725_060602/vis_data/,\ 
TrafoEnc_CamDom_OLD:/e/scratch/nxtaim-1/huber7/data/260617_dinov3_LoRA_V3/889927_ppp_dinov3_LoRA_V3_25k/20260615_231613/vis_data/,\ 
TrafoEnc_BEVpolDom:/e/scratch/nxtaim-1/huber7/work_dirs/1041092_ppp_finals_TrafoEnc_BEVpolDom_20k/20260725_060631/vis_data/,\ 
TrafoEncDec:/e/scratch/nxtaim-1/huber7/work_dirs/1041329_ppp_finals_TrafoEncDec_22k/20260725_072222/vis_data/,\ 
TrafoEncDec_woVel:/e/scratch/nxtaim-1/huber7/work_dirs/1041328_ppp_finals_TrafoEncDec_22k_woVel/20260725_072209/vis_data/,\ 
TrafoEncDec_woVel_woRCS:/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1046412_ppp_finals_TrafoEncDec_22k_woVel_woRCS/20260726_102253/vis_data/,\ 
TrafoEncDec_woVel_woRCS_OLD1:/p/scratch/nxtaim-1/huber7/data1/260627_ppp_dinov3_LoRA_V3_polar_liif/14061908_ppp_dinov3_LoRA_V3_polar_liif_12k/20260626_235547/vis_data/,\ 
TrafoEncDec_woVel_woRCS_OLD2:/p/scratch/nxtaim-1/huber7/data1/260627_ppp_dinov3_LoRA_V3_polar_liif/14063822_ppp_dinov3_LoRA_V3_polar_liif_12k/20260627_134840/vis_data/,\ 
TrafoEncDec_woVel_woRCS_OLD3:/p/scratch/nxtaim-1/huber7/data1/260627_ppp_dinov3/"\
    --port=12355 serve
    ;;
    9)
    tensorboard --logdir_spec="\
TrafoEnc_CamDom:/e/scratch/nxtaim-1/huber7/work_dirs/1041091_ppp_finals_TrafoEnc_CamDom_24k/20260725_060602/vis_data/,\ 
TrafoEnc_CamDom_OLD:/e/scratch/nxtaim-1/huber7/data/260617_dinov3_LoRA_V3/889927_ppp_dinov3_LoRA_V3_25k/20260615_231613/vis_data/,\ 
TrafoEnc_BEVpolDom:/e/scratch/nxtaim-1/huber7/work_dirs/1041092_ppp_finals_TrafoEnc_BEVpolDom_20k/20260725_060631/vis_data/"\
    --port=12355 serve
    ;;
  *)
    echo "Invalid option: $choice"
    echo "Usage: $0 [0-9]"
    exit 1
    ;;
esac