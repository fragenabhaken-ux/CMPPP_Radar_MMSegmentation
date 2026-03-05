module --force purge
module load Stages/2025  GCCcore/.13.3.0
module load tensorboard

choice=${1:-0}

case "$choice" in
  0)
    tensorboard --logdir_spec="\
50GBVRAM?:/p/scratch/nxtaim-1/huber7/work_dirs/mmsegmentation_ppp/20260223_194008/vis_data/" \
    --port=12345 serve
    ;;
  1)
    tensorboard --logdir_spec="\
50GBVRAM?:/p/scratch/nxtaim-1/huber7/work_dirs/mmsegmentation_ppp/20260223_194008/vis_data/" \
    --port=12346 serve
    ;;
  2)
    tensorboard --logdir_spec="\
50GBVRAM?:/p/scratch/nxtaim-1/huber7/work_dirs/mmsegmentation_ppp/20260223_201606/vis_data/" \
    --port=12346 serve
    ;;
  3)
    tensorboard --logdir_spec="\
50GBVRAM?:/p/scratch/nxtaim-1/huber7/work_dirs/mmsegmentation_ppp/20260223_205728/vis_data/" \
    --port=12346 serve
    ;;

  *)
    echo "Invalid option: $choice"
    echo "Usage: $0 [0-8]"
    exit 1
    ;;
esac