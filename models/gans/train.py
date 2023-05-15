import sys

sys.path.append('/home/v_chernyy/thesis')


from torchvision import transforms as TT
from data import get_train_test_split_paired_loaders

from models.gans.cycle_gan.model import GAN

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import pytorch_lightning as pl

import yaml

with open('configs/train_gan.yaml') as f:
    params_dict = yaml.safe_load(f)
    
print(params_dict)

train_transform = TT.Resize((256, 256))

train_loader, test_loader = get_train_test_split_paired_loaders(
    params_dict['data'],
    train_transform=train_transform,
    test_transform=train_transform
)

model = GAN(
    **params_dict['model']
)

d_from = params_dict['data']['df_from']['data_path'].split('/')[-1].split('.')[0]
d_to = params_dict['data']['df_to']['data_path'].split('/')[-1].split('.')[0]

wb_logger = pl.loggers.WandbLogger(
    name='{}_BS={}| N_EPOCHS={}_{}'.format(
        f'{d_from}->{d_to}_gan',
        params_dict['data']['batch_size'],
        params_dict['trainer']['max_epochs'],
        params_dict['version']
        ),
    project='thesis',
    log_model='all'
)

checkpoint_callback = ModelCheckpoint(
    filename='{}'.format(
        f'{d_from}->{d_to}_gan'
    ),
    **params_dict['checkpoint']
)

# es_callback = EarlyStopping(
#     monitor='val Surface Dice',
#     min_delta=5.,
#     patience=20,
#     mode='max'
# )

lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(
    logger=wb_logger,
    **params_dict['trainer'],
    callbacks=[checkpoint_callback, lr_monitor]#, es_callback]
)

trainer.fit(
    model = model,
    train_dataloaders = train_loader,
    val_dataloaders = test_loader
)



