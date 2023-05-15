# from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from PIL import Image
from torchvision import transforms as TT
import torchvision.transforms.functional as TF

class SegmentationDataset(Dataset):
    def __init__(self, dataframe, transform=None, mask_transform=None, mode='train'):
        super().__init__()

        self.transform = transform
        self.mask_transform = mask_transform

        self.df = dataframe.reset_index()
        self.mode = mode

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, index):
        img = Image.open(self.df.iloc[index]['img'])
        mask = Image.open(self.df.iloc[index]['mask'])

        img = TT.ToTensor()(img)
        mask = (TT.PILToTensor()(mask) / 255).long()

        # if self.mode == 'train':
        #     i, j, h, w = TT.RandomCrop.get_params(
        #         img, output_size=(160, 160))
        #     img = TF.crop(img, i, j, h, w)
        #     mask = TF.crop(mask, i, j, h, w)
            
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        return {'img': img, 'mask': mask}
    

def get_train_test_split_loaders(data_dict, train_transform=None, test_transform=None):

    df = pd.read_csv(data_dict['data_path'])

    train_subset = SegmentationDataset(
        df[df['fold'] != 'test'],
        transform=train_transform,
        mode='train'
    )

    test_subset = SegmentationDataset(
        df[df['fold'] == 'test'],
        transform=test_transform,
        mode='test'
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=data_dict['batch_size'], 
        pin_memory=True,
        num_workers=40,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=1, 
        pin_memory=True,
        num_workers=40
    )

    return train_loader, test_loader
    
class DADataset(Dataset):
    def __init__(self, df_from_dict, df_to_dict, is_test=False, transform=None, mask_transform=None):
        super().__init__()

        self.transform = transform
        self.mask_transform = mask_transform

        df_from = pd.read_csv(df_from_dict['data_path'])
        df_to = pd.read_csv(df_to_dict['data_path'])

        self.data_from = SegmentationDataset(
            df_from[(df_from['fold'] != 'test') if not is_test else (df_from['fold'] == 'test')],
            transform=transform
        )

        self.data_to = SegmentationDataset(
            df_to[(df_to['fold'] != 'test') if not is_test else (df_to['fold'] == 'test')],
            transform=transform
        )

    def __len__(self):
        return min(len(self.data_from), len(self.data_to))
        
    def __getitem__(self, index):
        return self.data_from[index], self.data_to[index]
    
def get_train_test_split_paired_loaders(data_dict, train_transform=None, test_transform=None):

    train_dataset_AB = DADataset(
        data_dict['df_from'],
        data_dict['df_to'],
        transform=train_transform,
        is_test=False
    )

    test_dataset_AB = DADataset(
        data_dict['df_from'],
        data_dict['df_to'],
        transform=test_transform,
        is_test=True
    )

    train_loader_AB = DataLoader(
        train_dataset_AB,
        batch_size=data_dict['batch_size'], 
        pin_memory=True,
        num_workers=40,
        shuffle=True
    )
    
    test_loader_AB = DataLoader(
        test_dataset_AB,
        batch_size=1, 
        pin_memory=True,
        num_workers=40
    )

    return train_loader_AB, test_loader_AB
