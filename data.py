from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from PIL import Image

class SegmentationDataset(Dataset):
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
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.mask_transform is not None:
            mask = self.mask_transform(mask).long()

        return {'img': img, 'mask': mask}
    

def get_train_test_split_loaders(dataset, batch_size, test_size):

    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        shuffle=False#True
    )

    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size, 
        pin_memory=True,
        num_workers=40,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size, 
        pin_memory=True,
        num_workers=40
    )

    return train_loader, test_loader