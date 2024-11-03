from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class LAKEDataset_Group6(CustomDataset):
    """LAKEDataset dataset.

    """

# 0 background: sky
# 1 Stable: Concrete, Asphalt 
# 2 Granular: Sand, Dirt, mulch, gravel, water
# 3 Poor foothold: rocks, rockbed
# 4 High resistance: grass, bush, log
# 5 Obstacle: tree, pole, vehicle, container/generic object, bicycle, person, fence, sign, bridge, table, building

    CLASSES = ("background", "smooth", "rough", "bumpy", "forbidden", "obstacle")

    PALETTE = [[ 0, 0, 0 ], [ 0,128,0 ],[ 255, 255, 0 ],[ 255, 128, 0 ],
            [ 255, 0, 0 ],[  0, 0, 128] ]


    def __init__(self, **kwargs):
        super(LAKEDataset_Group6, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_group6.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        self.CLASSES = ("background", "smooth", "rough", "bumpy", "forbidden", "obstacle")
        self.PALETTE = [[ 0, 0, 0 ], [ 0,128,0 ],[ 255, 255, 0 ],[ 255, 128, 0 ],
            [ 255, 0, 0 ],[  0, 0, 128] ]
        # assert osp.exists(self.img_dir)