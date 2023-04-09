from torchvision import transforms as TT
from data import get_train_test_split_loaders

from models.seg_wrapper import LitSegNet
from models.unet import UNet2D

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import yaml


with open('configs/train_unet.yaml') as f:
    params_dict = yaml.safe_load(f)
    
print(params_dict)

# train_transform = (
#     TT.Compose([
#         # TT.RandomChoice([
#         #     TT.Compose([
#         #         TT.Resize(256),
#         #         TT.CenterCrop(256)
#         #     ]),
#         #     TT.Resize((256, 256)),
#         # ])
#         TT.Resize((256, 256)),
#     ])
# )

# test_transform = None #TT.Compose([
#    TT.Resize((256, 256)),
#])

# SIEMENS_PATH

# PHILIPS_PATH
# dataset_B = SegmentationDataset(
#     dataframe_path = params_dict['tgt_path'],
#     transform=transforms,
#     mask_transform=transforms
# )

trainA_loader, testA_loader = get_train_test_split_loaders(
    params_dict['data'],
    # train_transform=train_transform,
    # test_transform=None
)

print(params_dict['data']['batch_size'])

# trainB_loader, testB_loader = get_train_test_split_loaders(
#     dataset_B, batch_size=params_dict['data']['batch_size'], test_size=0.2
# )

model = LitSegNet(
    n_epochs=params_dict['trainer']['max_epochs'],
    num_classes=1,
    backbone=UNet2D(
        **params_dict['model']
    )
)

domain = params_dict['data']['data_path'].split('/')[-1].split('.')[0]

wb_logger = pl.loggers.WandbLogger(
    name='{}_BS={}| N_EPOCHS={}'.format(
        f'{domain}_unet',
        params_dict['data']['batch_size'],
        params_dict['trainer']['max_epochs']
        ),
    project='thesis',
    log_model='all'
)

checkpoint_callback = ModelCheckpoint(
    filename='{}'.format(
        f'{domain}_unet'
    ),
    **params_dict['checkpoint']
)

es_callback = EarlyStopping(
    monitor='val Surface Dice',
    min_delta=5.,
    patience=20,
    mode='max'
)

trainer = pl.Trainer(
    logger=wb_logger,
    **params_dict['trainer'],
    callbacks=[checkpoint_callback, es_callback]
)

trainer.fit(
    model = model,
    train_dataloaders = trainA_loader,
    val_dataloaders = testA_loader, 
)