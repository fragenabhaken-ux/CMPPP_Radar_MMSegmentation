module --force purge
module load Stages/2025  GCCcore/.13.3.0
module load tensorboard

choice=${1:-0}

case "$choice" in
  0)
    tensorboard --logdir_spec="\
dinov3_V2_1:/p/scratch/nxtaim-1/huber7/work_dirs/13941945_ppp_dinov3_LoRA_rope_big_neck_noCrop_V2_12k/20260606_104319/vis_data/,\
dinov3_V2_2:/p/scratch/nxtaim-1/huber7/work_dirs/13942174_ppp_dinov3_LoRA_rope_big_neck_noCrop_V2_12k/20260606_131019/vis_data/,\
dinov3_V2_3:/p/scratch/nxtaim-1/huber7/work_dirs/13942624_ppp_dinov3_LoRA_rope_big_neck_noCrop_V2_12k/20260606_163253/vis_data/,\
dinov3_V2_4:/p/scratch/nxtaim-1/huber7/work_dirs/13943679_ppp_dinov3_LoRA_rope_big_neck_noCrop_V2_12k/20260606_202059/vis_data/"\
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
dinov3_polar_BEV:/p/scratch/nxtaim-1/huber7/work_dirs/14063822_ppp_dinov3_LoRA_V3_polar_liif_12k/20260627_134840/vis_data/" \
    --port=12303 serve
    ;;
  3)
    tensorboard --logdir_spec="\
dinov3_lora:/p/scratch/nxtaim-1/huber7/work_dirs/13900238_ppp_dinov3_LoRA_rope_big_neck_7k_testrun/20260529_200721/vis_data/" \
    --port=12305 serve
    ;;
  4)
  tensorboard --logdir_spec="\
dino_big_LoRA_P1:/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13919691_ppp_dinov3_big_LoRA_rope_big_neck_noCrop_12k_P1/20260603_195002/vis_data/,\
dino_big_LoRA_P2:/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13920974_ppp_dinov3_big_LoRA_rope_big_neck_noCrop_12k_P2/20260603_233854/vis_data/" \
  --port=12355 serve
  ;;
  5)
  tensorboard --logdir_spec="\
dinov3_base_P1:/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13909632_ppp_dinov3_rope_big_neck_noCrop_12k_P1/20260601_225009/vis_data/,\
dinov3_base_P2:/p/scratch/nxtaim-1/huber7/work_dirs/13925792_ppp_dinov3_rope_big_neck_noCrop_12k_P2/20260604_150719/vis_data/" \
  --port=12304 serve
  ;;
  6)
  tensorboard --logdir_spec="\
dinov3_multitask_P1:/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13914010_ppp_dinov3_rope_big_neck_depthLoss_multitask_head_noCrop_12k_P1/20260602_165648/vis_data/,\
dinov3_multitask_P2:/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13914735_ppp_dinov3_rope_big_neck_depthLoss_multitask_head_noCrop_12k_P2/20260602_223946/vis_data/" \
    --port=12305 serve
    ;;
  7)
  tensorboard --logdir_spec="\
dinov3_multitask_two_depth_P1:/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13917929_ppp_dinov3_rope_big_neck_depthLoss_multitask_two_depth_head_noCrop_12k_P1/20260603_142924/vis_data/,\
dinov3_multitask_two_depth_P2:/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13918698_ppp_dinov3_rope_big_neck_depthLoss_multitask_two_depth_head_noCrop_12k_P2/20260603_160635/vis_data/" \
    --port=12301 serve
    ;;
  8)
  tensorboard --logdir_spec="\
dino_LoRA_P1:/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13917932_ppp_dinov3_LoRA_rope_big_neck_noCrop_12k_P1/20260603_143025/vis_data/,\
dino_LoRA_P2:/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13918853_ppp_dinov3_LoRA_rope_big_neck_noCrop_12k_P2/20260603_164558/vis_data/" \
  --port=12302 serve
  ;;
  9)
  tensorboard --logdir_spec="\
dino_mid_LoRA_P1:/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13919690_ppp_dinov3_mid_LoRA_rope_big_neck_noCrop_12k_P1/20260603_194852/vis_data/,\
dino_mid_LoRA_P2:/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13920972_ppp_dinov3_mid_LoRA_rope_big_neck_noCrop_12k_P2/20260603_233759/vis_data/" \
    --port=12303 serve
  ;;

  *)
    echo "Invalid option: $choice"
    echo "Usage: $0 [0-9]"
    exit 1
    ;;
esac