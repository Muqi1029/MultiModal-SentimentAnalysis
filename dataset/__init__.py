from torch.utils.data import Dataset
import os
from PIL import Image


class MultiModalDataset(Dataset):
    def __init__(self, imgs_dir, text_dir, labels, guids, config=None):
        super().__init__()
        self.imgs_dir = imgs_dir
        self.text_dir = text_dir
        self.labels = labels
        self.guids = guids

    def __len__(self):
        return len(self.guids)

    def __getitem__(self, idx):
        guid = str(self.guids[idx])
        label = self.labels[idx]
        img = Image.open(os.path.join(self.imgs_dir, guid + ".jpg"))

        text = open(os.path.join(self.text_dir, guid + ".txt"), errors="ignore").read()
        text = '[CLS]' + text.replace("#", "") + '[SEP]'

        return guid, img, text, label


