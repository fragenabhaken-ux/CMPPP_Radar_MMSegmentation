import mmcv
from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms.loading import BaseTransform


@TRANSFORMS.register_module()
class LoadInstanceAnnotations(BaseTransform):

    def transform(self, results):
        seg_path = results['seg_map_path']

        # Replace labelIds → instanceIds
        inst_path = seg_path.replace(
            '_gtFine_labelIds.png',
            '_gtFine_instanceIds.png'
        )

        inst_mask = mmcv.imread(inst_path, flag='unchanged')

        results['gt_seg_map'] = inst_mask
        return results
