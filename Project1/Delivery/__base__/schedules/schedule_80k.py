classes = ["background","skin","nose","eye_glasses","left_eye","right_eye",
           "left_brow","right_brow","left_ear","right_ear","mouth","upper_lip",
           "lower_lip","hair","hat","earing","necklace","neck","cloth"]   
palette = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000, meta=dict(CLASSES=classes, PALETTE=palette))
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)

gpu_ids = range(0,1)
seed = 0