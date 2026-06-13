module --force purge
module load Stages/2025  GCCcore/.13.3.0
module load tensorboard

choice=${1:-0}

case "$choice" in
  0)
    tensorboard --logdir_spec="\
TEST_JUPITER:/e/scratch/nxtaim-1/huber7/work_dirs/852692_ppp_dinov3_rope_big_neck_40k_JUPITER_TEST/20260612_135750/vis_data/"\
    --port=12355 serve
    ;;
  1)
    tensorboard --logdir_spec="\
unet:/p/scratch/nxtaim-1/huber7/work_dirs/13763261_unet_160k/20260425_004229/vis_data/,\
unet_8nodes:/p/scratch/nxtaim-1/huber7/work_dirs/13767623_unet_160k_8nodes/20260427_013913/vis_data/" \
    --port=12301 serve
    ;;
  *)
    echo "Invalid option: $choice"
    echo "Usage: $0 [0-9]"
    exit 1
    ;;
esac