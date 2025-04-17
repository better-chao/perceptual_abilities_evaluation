_base_ = './glip-A_lvis_tp.py'

model = dict(bbox_head=dict(early_fuse=True))
