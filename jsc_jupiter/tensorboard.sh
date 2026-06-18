module --force purge
module load Stages/2025  GCCcore/.13.3.0
module load tensorboard

choice=${1:-0}

case "$choice" in
  0)
    tensorboard --logdir_spec="\
unet_init_test:/e/scratch/nxtaim-1/huber7/work_dirs/858021_ppp_unet_5k_init_test/20260613_175308/vis_data/"\
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
dinov3_LoRA_V3_disparity:/e/scratch/nxtaim-1/huber7/data/260617_dinov3_LoRA_V3/906347_ppp_dinov3_LoRA_V3_disparity_25k/20260617_043344/vis_data,\
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
  *)
    echo "Invalid option: $choice"
    echo "Usage: $0 [0-9]"
    exit 1
    ;;
esac