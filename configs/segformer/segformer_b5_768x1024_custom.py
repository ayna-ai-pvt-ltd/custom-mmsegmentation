_base_ = [
    '../_base_/default_runtime.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'

# Model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(1024, 768))

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MixVisionTransformer',
        with_cp=True,
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint)),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=768,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_weight=3.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# Dataset settings
dataset_type = 'CustomDataset'
data_root = '/home/ubuntu/ayna_disk_1/final_dataset/final_combined_dataset'

crop_size = (1024, 768)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='Resize', scale=(1024, 768), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(1024, 768), keep_ratio=False),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/images',
            seg_map_path='train/label-masks'),
        pipeline=train_pipeline
    ))

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/new-images',
            seg_map_path='val/label-masks'),
        pipeline=test_pipeline
    ))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], pre_eval=True)
test_evaluator = val_evaluator

# Optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    loss_scale='dynamic')

# Learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=250000,
        by_epoch=False)
]

# Training settings
train_cfg = dict(
    type='IterBasedTrainLoop', 
    max_iters=250000, 
    val_interval=200)
val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='segformer-finetuning',
            entity='ankitkumardos47-indian-indian-institute-of-information-t',
            name='cont_1_segformer_330k_1e-6',
        ),
        define_metric_cfg={'mIoU': 'max', 'loss': 'min'}
    )
]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='SegVisualizationHook',
        draw=True,
        interval=10,
        show=False,
        wait_time=0
    )
)

# Set random seed
randomness = dict(seed=42, deterministic=False)

# Enable automatic mixed precision training
fp16 = dict(loss_scale='dynamic')

# Logging
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=False)

# Work directory
work_dir = '/home/ubuntu/ayna_disk_0/segformer/cont_segformer_330k_1e-6'

# Load from checkpoint (optional)
load_from = None
resume = False