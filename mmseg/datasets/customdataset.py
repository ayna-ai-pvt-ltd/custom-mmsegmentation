# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CustomDataset(BaseSegDataset):
    """Custom dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png' for Custom dataset.
    """
    METAINFO = dict(
        classes=('background', 'headwear', 'hair', 'gloves', 'glasses',
            'coat', 'dress', 'outerwear', 'socks', 'pants',
            'skin', 'scarf', 'skirt', 'face', 'shoes',
            'bag', 'accessory', 'jumpsuit', 'stand'),
        palette=[
            [192, 192, 192],  # background - #C0C0C0
            [245, 63, 63],    # headwear - #F53F3F
            [120, 22, 255],   # hair - #7816FF
            [0, 180, 42],     # gloves - #00B42A
            [22, 93, 255],    # glasses - #165DFF
            [255, 125, 0],    # coat - #FF7D00
            [235, 10, 164],   # dress - #EB0AA4
            [123, 198, 22],   # outerwear - #7BC616
            [134, 140, 156],  # socks - #868C9C
            [219, 158, 0],    # pants - #DB9E00
            [183, 29, 232],   # skin - #B71DE8
            [15, 198, 194],   # scarf - #0FC6C2
            [255, 180, 0],    # skirt - #FFB400
            [22, 141, 253],   # face - #168DFD
            [255, 87, 34],    # shoes - #FF5722
            [255, 22, 162],   # bag - #FF16A2
            [22, 255, 185],   # accessory - #16FFB9
            [247, 17, 212],   # jumpsuit - #F711D4
            [212, 87, 106]    # stand - #D4576A
        ])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
