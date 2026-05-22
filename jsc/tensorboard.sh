module --force purge
module load Stages/2025  GCCcore/.13.3.0
module load tensorboard

choice=${1:-0}

case "$choice" in
  0)
    tensorboard --logdir_spec="\
unet:/p/scratch/nxtaim-1/huber7/work_dirs/13763261_unet_160k/20260425_004229/vis_data/,\
upernet_r50:/p/scratch/nxtaim-1/huber7/work_dirs/13763638_upernet_r50_160k/20260425_093832/vis_data/,\
upernet_r50pretrained:/p/scratch/nxtaim-1/huber7/work_dirs/13763640_upernet_r50pretrained_160k/20260425_093833/vis_data/,\
resnet34_fcn:/p/scratch/nxtaim-1/huber7/work_dirs/13765446_resnet34_fcn_160k/20260426_003026/vis_data/" \
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
resnet34_fcn:/p/scratch/nxtaim-1/huber7/work_dirs/13767582_resnet34_fcn_160k_8nodes/20260427_011232/vis_data/" \
    --port=12357 serve
    ;;
  3)
    tensorboard --logdir_spec="\
unet_8nodes:/p/scratch/nxtaim-1/huber7/work_dirs/13767623_unet_160k_8nodes/20260427_013913/vis_data/" \
    --port=12300 serve
    ;;
  4)
  tensorboard --logdir_spec="\
unet:/p/scratch/nxtaim-1/huber7/work_dirs/13763261_unet_160k/20260425_004229/vis_data/" \
  --port=12300 serve
  ;;
  5)
  tensorboard --logdir_spec="\
dino:/p/scratch/nxtaim-1/huber7/data1/260519_dinov3_5k_LR/13863109_dinov3_LRRangeTest/20260518_224130/vis_data/" \
  --port=12300 serve
  ;;

  *)
    echo "Invalid option: $choice"
    echo "Usage: $0 [0-8]"
    exit 1
    ;;
esac