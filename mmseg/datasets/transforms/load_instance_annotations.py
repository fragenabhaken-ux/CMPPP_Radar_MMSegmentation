import mmcv
import numpy as np
from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms.loading import BaseTransform


@TRANSFORMS.register_module()
class LoadInstanceAnnotations(BaseTransform):

    def transform(self, results):
        seg_path = results['seg_map_path']

        # labelIds → instanceIds Pfad
        inst_path = seg_path.replace(
            '_gtFine_labelIds.png',
            '_gtFine_instanceIds.png'
        )

        inst_mask = mmcv.imread(inst_path, flag='unchanged')

        # Sicherstellen: 2D (H, W), kein unnötiger Kanal
        if inst_mask.ndim == 3:
            inst_mask = inst_mask[..., 0]

        # Wie bei LoadAnnotations:
        results['gt_seg_map'] = inst_mask.astype(np.int32)

        # GANZ WICHTIG: seg_fields registrieren,
        # damit RandomCrop / Resize / Flip auch gt_seg_map anfassen.
        if 'seg_fields' not in results:
            results['seg_fields'] = []
        if 'gt_seg_map' not in results['seg_fields']:
            results['seg_fields'].append('gt_seg_map')

        return results
