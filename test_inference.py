from mmseg.apis import init_model, inference_model
from mmseg.visualization import SegLocalVisualizer
from mmengine.structures import PixelData
import torch
import os
import cv2

# ---- paths ----
config_file = 'configs/fastscnn/fast_scnn_8xb4-160k_cityscapes-512x1024.py'
checkpoint_file = (
    'checkpoints/'
    'fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth'
)
image_file = 'demo/demo.png'
output_file = 'test_inference_output.png'

if torch.cuda.is_available():
    print("CUDA is available")
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0))
    device = 'cuda:0'
else:
    print("CUDA is NOT available, using CPU")
    device = 'cpu'

# ---- init model ----
model = init_model(
    config=config_file,
    checkpoint=checkpoint_file,
    device=device
)

# ---- inference ----
result = inference_model(model, image_file)

pred = result.pred_sem_seg.data
print("pred type:", type(pred))
print("pred shape:", pred.shape)
print("pred unique labels:", torch.unique(pred))

sem_seg = result.pred_sem_seg.data

# ensure shape (H, W)
if sem_seg.ndim == 3:
    sem_seg = sem_seg[0]

# wrap back into PixelData
result.pred_sem_seg = PixelData(data=sem_seg.unsqueeze(0))

mask = result.pred_sem_seg.data[0].cpu().numpy().astype('uint8')
cv2.imwrite("mask.png", mask)

# ---- visualize & save ----
visualizer = SegLocalVisualizer(
    vis_backends=[dict(type='LocalVisBackend')],
    save_dir='.'
)

visualizer.dataset_meta = model.dataset_meta
visualizer.add_datasample(
    name='result',
    image=cv2.imread(image_file)[:, :, ::-1],
    data_sample=result,
    show=False,
    out_file=output_file
)

print("Inference finished")
print("Saved:", os.path.abspath(output_file))