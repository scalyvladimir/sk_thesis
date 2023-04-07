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

        if self.mode == 'train':
            i, j, h, w = TT.RandomCrop.get_params(
                img, output_size=(160, 160))
            img = TF.crop(img, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        return {'img': img, 'mask': mask}
    

def get_train_test_split_loaders(data_dict, train_transform=None, test_transform=None):

    df = pd.read_csv(data_dict['data_path'])

    train_subset = SegmentationDataset(
        df[~df['file_id'].isin(data_dict['exclude_scans_list'])],
        transform=train_transform,
        mode='train'
    )

    test_subset = SegmentationDataset(
        df[df['file_id'].isin(data_dict['exclude_scans_list'])],
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
    
class SegmentationPseudoDataset(Dataset):
    def __init__(self, dataframe_path, transform=None, mask_transform=None):
        super().__init__()

        self.transform = transform
        self.mask_transform = mask_transform

        self.df = pd.read_csv(dataframe_path)

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, index):
        img = Image.open(self.df.iloc[index]['img'])
        mask = Image.open(self.df.iloc[index]['mask'])
        ps_mask = Image.open(self.df.iloc[index]['pseudo_mask'])
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.mask_transform is not None:
            mask = self.mask_transform(mask).long()
            ps_mask = self.mask_transform(ps_mask).long()

        return {'img': img, 'mask': mask, 'pseudo_mask': ps_mask}