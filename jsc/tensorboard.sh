module --force purge
module load Stages/2025  GCCcore/.13.3.0
module load tensorboard

choice=${1:-0}

case "$choice" in
  0)
    tensorboard --logdir_spec="\
50GBVRAM?:/p/scratch/nxtaim-1/huber7/work_dirs/13699766_LRRangeTest_clipNorm_100/20260414_154437/vis_data/" \
    --port=12345 serve
    ;;
  1)
    tensorboard --logdir_spec="\
50GBVRAM?:/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13493442_clipNorm_1/20260318_202515/vis_data/" \
    --port=12345 serve
    ;;
  2)
    tensorboard --logdir_spec="\
50GBVRAM?:/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13493459_clipNorm_10/20260318_203220/vis_data/" \
    --port=12345 serve
    ;;
  3)
    tensorboard --logdir_spec="\
50GBVRAM?:/p/data1/nxtaim/users/huber7/models/20260312/mmsegmentation_ppp/20260312_052718/vis_data/" \
    --port=12345 serve
    ;;

  *)
    echo "Invalid option: $choice"
    echo "Usage: $0 [0-8]"
    exit 1
    ;;
esac