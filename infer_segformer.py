from torchvision import transforms as TT
from data import SegmentationDataset

from models.seg_wrapper import LitSegNet

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from models.utils import pad_tensor, unpad_tensor
import infer_utils

import pandas as pd
import numpy as np

import torch

from tqdm import tqdm

import yaml
import os

with open('configs/infer_segformer.yaml') as f:
    params_dict = yaml.safe_load(f)
    
# print(params_dict)

test_transform = None

df = pd.read_csv(params_dict['data']['data_path'])

test_subset = SegmentationDataset(
    df[df['fold'] == 'test'],
    transform=test_transform,
    mode='test'
)

test_loader = DataLoader(
    test_subset,
    batch_size=1, 
    pin_memory=True,
    num_workers=40
)

DEVICE = torch.device('cuda:{}'.format(params_dict['gpu_id']) if torch.cuda.is_available() else 'cpu')
model = LitSegNet.load_from_checkpoint(checkpoint_path=params_dict['model']['checkpoint_path']).to(DEVICE).eval()

with torch.no_grad():

    avg_sds = 0.

    for batch in tqdm(test_loader):

        x = torch.tensor(pad_tensor(batch['img'].cpu())).to(DEVICE)

        out = unpad_tensor(model(x), batch['img']).squeeze(1)
        pred_mask = (torch.sigmoid(out) > 0.5).long()
        src_mask = batch['mask'].squeeze(1)

        # print(pred_mask.shape, src_mask.shape)

        sds_metric = np.mean([model.surface_dice_score(x, y) for x, y in zip(pred_mask, src_mask)])
        avg_sds += sds_metric

        # print(sds_metric)

    avg_sds = avg_sds / len(test_loader)

    print('\nAverage Surface Dice Score:', avg_sds)

    os.makedirs('inference_results/', exist_ok=True)

    chkpt_name = params_dict['model']['checkpoint_path'].split('/')[-1].split('.')[0]

    df_from = chkpt_name.split('_')[0]
    df_to = params_dict['data']['data_path'].split('/')[-1].split('.')[0]

    df_path = 'inference_results/{}.csv'.format(params_dict['data']['save_filename'])

    if os.path.isfile(df_path):
        df = pd.read_csv(df_path, index_col=0)
    else:
        df = infer_utils.create_df()

    print(df_from, df_to)

    if df_from not in df.index:
        df_new_row = pd.DataFrame(pd.Series({df_from: avg_sds}, name=df_to), columns=df.columns)
        df = pd.concat([df, df_new_row])
    else:
        df.loc[df.index == df_from, df_to] = avg_sds
    
    df.to_csv(df_path)