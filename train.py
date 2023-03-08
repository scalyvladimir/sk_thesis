from torchvision import transforms as TT
from data import SegmentationDataset, get_train_test_split_loaders

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.supporters import CombinedLoader

from models.supervised_models import LitDeepLabV2

import yaml


with open('configs/train.yaml') as f:
    params_dict = yaml.safe_load(f)
    
print(params_dict)

transforms = TT.Compose([
    TT.ToTensor()
])

# SIEMENS_PATH
dataset_A = SegmentationDataset(
    dataframe_path = params_dict['src_path'],
    transform=transforms,
    mask_transform=transforms
)

# PHILIPS_PATH
dataset_B = SegmentationDataset(
    dataframe_path = params_dict['tgt_path'],
    transform=transforms,
    mask_transform=transforms
)

trainA_loader, testA_loader = get_train_test_split_loaders(
    dataset_A, batch_size=params_dict['data']['batch_size'], test_size=0.2
)

trainB_loader, testB_loader = get_train_test_split_loaders(
    dataset_B, batch_size=params_dict['data']['batch_size'], test_size=0.2
)

model = LitDeepLabV2(
    n_epochs=params_dict['n_epochs'],
    **params_dict['model']
)

wb_logger = pl.loggers.WandbLogger(
    name='{}_BS={}| N_EPOCHS={}| beta={:.2f}'.format(
        params_dict['logger']['name'],
        params_dict['data']['batch_size'],
        params_dict['n_epochs'],
        params_dict['model']['beta']
        ),
    project=params_dict['logger']['project']
)

checkpoint_callback = ModelCheckpoint(
    filename='{}_beta={:.2f}'.format(
        params_dict['logger']['name'],
        params_dict['model']['beta'],
    ),
    **params_dict['checkpoint']
)

trainer = pl.Trainer(
    max_epochs=params_dict['n_epochs'],
    logger=wb_logger,
    accelerator='gpu',
    devices=params_dict['gpu_id'],
    benchmark=True,
    callbacks=[checkpoint_callback]
)

trainer.fit(
    model = model,
    train_dataloaders = CombinedLoader({
        'src': trainA_loader,
        'tgt': trainB_loader
    }),
    val_dataloaders = CombinedLoader({
        'src': testA_loader,
        'tgt': testB_loader
    })
)