import numpy as np
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class MiddleBandRandomHorizontalCrop:
    """
    Crops a horizontal window of width `crop_w` at a random horizontal
    position, but always keeps a fixed vertical middle band of height `crop_h`.

    This avoids cropping the sky (top) and the pure road region (bottom),
    ensuring that the crop always contains street objects like cars/persons.

    Args:
        crop_size (tuple): (crop_h, crop_w)
        vertical_band_ratio (tuple): (lower_frac, upper_frac)
            Example: (0.3, 0.7) → keep 30%-70% of image height
    """

    def __init__(self, crop_size=(512, 1024), vertical_band_ratio=(0.3, 0.7)):
        self.crop_h, self.crop_w = crop_size
        self.lower_frac, self.upper_frac = vertical_band_ratio

    def __call__(self, results):

        img = results["img"]
        H, W = img.shape[:2]

        # --- Vertical placement: fixed middle band ---
        y_low = int(H * self.lower_frac)
        y_high = int(H * self.upper_frac)

        # Clamp crop to that region
        if self.crop_h >= (y_high - y_low):
            # If crop_h is too large, force center alignment
            y1 = max(0, (H - self.crop_h) // 2)
        else:
            span = (y_high - self.crop_h) - y_low
            y1 = y_low + (np.random.randint(span + 1) if span > 0 else 0)

        y2 = y1 + self.crop_h

        # --- Horizontal placement: random ---
        if self.crop_w >= W:
            x1 = 0
        else:
            max_shift = W - self.crop_w
            x1 = np.random.randint(max_shift + 1)

        x2 = x1 + self.crop_w

        # --- Crop the image ---
        results['img'] = img[y1:y2, x1:x2, :]

        # --- Crop all annotation maps ---
        for key in ['gt_seg_map', 'gt_semantic_seg',
                    'gt_panoptic_seg', 'gt_instance_map']:
            if key in results:
                results[key] = results[key][y1:y2, x1:x2]

        # Update shape
        results['img_shape'] = (self.crop_h, self.crop_w)
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(crop_size={self.crop_h, self.crop_w}, "
                f"vertical_band_ratio={self.lower_frac, self.upper_frac})")
