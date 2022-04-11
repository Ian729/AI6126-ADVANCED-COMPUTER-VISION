_base_ = [
    './upernet_swin_tiny_patch4_window7_512x512_160k_pretrain_224x224_1K.py'
]
model = dict(
    pretrained='./swin_base/swin_base_patch4_window7_224.pth',
    backbone=dict(
        embed_dims=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=19),
    auxiliary_head=dict(in_channels=512, num_classes=19))