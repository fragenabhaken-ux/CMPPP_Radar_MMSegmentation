module --force purge
module load Stages/2025  GCCcore/.13.3.0
module load tensorboard

choice=${1:-0}

case "$choice" in
  0)
    tensorboard --logdir_spec="\
50GBVRAM?:/p/scratch/nxtaim-1/huber7/work_dirs/13763261_unet_160k/20260425_004229/vis_data/" \
    --port=12355 serve
    ;;
  1)
    tensorboard --logdir_spec="\
50GBVRAM?:/p/scratch/nxtaim-1/huber7/work_dirs/13763638_upernet_r50_160k/20260425_093832/vis_data/" \
    --port=12356 serve
    ;;
  2)
    tensorboard --logdir_spec="\
50GBVRAM?:/p/scratch/nxtaim-1/huber7/work_dirs/13763640_upernet_r50pretrained_160k/20260425_093833/vis_data/" \
    --port=12357 serve
    ;;
  3)
    tensorboard --logdir_spec="\
50GBVRAM?:/p/scratch/nxtaim-1/huber7/work_dirs/13761661_LR_test/20260424_144703/vis_data/" \
    --port=12355 serve
    ;;

  *)
    echo "Invalid option: $choice"
    echo "Usage: $0 [0-8]"
    exit 1
    ;;
esac