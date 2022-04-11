from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmcv import Config
from mmseg.datasets import CelebAMaskDataset

import os.path as osp
import numpy as np
from PIL import Image
import mmcv


config_file = "./ACV/upernet_swin_base_512x512_160k.py"
checkpoint_file = "./swin_base/latest.pth"
cfg = Config.fromfile(config_file)

cfg.work_dir = "./swin_base/"
datasets = [build_dataset(cfg.data.train)]


if osp.isfile(checkpoint_file):
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
else:
    # Build the detector
    model = build_segmentor(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

model.init_weights()
model.CLASSES = datasets[0].CLASSES
model.PALETTE = datasets[0].PALETTE
# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True)