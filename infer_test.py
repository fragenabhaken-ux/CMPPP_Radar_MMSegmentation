from mmseg.apis import MMSegInferencer

# Modell + Checkpoint automatisch herunterladen
inferencer = MMSegInferencer(
    model='deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024'
)

# Inference auf Beispielbild
result = inferencer(
    'demo/demo.png',
    show=True,             # Segmentation visuell anzeigen
    wait_time=0            # Fenster offen lassen
)

print(result)