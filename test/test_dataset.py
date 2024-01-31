import sys
sys.path.append('../')
import os
import pytest
from main import config
from utils import get_dataloader
from torchvision.transforms import Compose, Resize, ToTensor
import json
from dataset import MultiModalDataset


class TestDataset:
    @pytest.fixture(autouse=True)
    def pre_test_dataset(self):
        self.config = config
        self.config['data_dir'] = "../data"
        self.config['train_path'] = "../preprocess/train.json"

    def test_dataset(self):
        data = json.load(open(self.config['train_path'], "r", encoding="utf-8"))
        guid = list(d['guid'] for d in data)
        labels = list(d['label'] for d in data)
        img_transform = Compose([
            Resize((224, 224)), ToTensor()]
        )
        multi_modal_dataset = MultiModalDataset(imgs_dir=self.config['data_dir'], text_dir=self.config['data_dir'],
                                                labels=labels, guids=guid, img_transform=img_transform, config=config)
        print(multi_modal_dataset[0])

    def test_data_loader(self):
        config['val_path'] = "../preprocess/val.json",
        config['train_path'] = "../preprocess/train.json"
        config['data_path'] = "../data/"
        data_loader = get_dataloader(config, "train_path")
        print("len(data_loader):", len(data_loader))
        for padded_texts, padded_texts_mask, imgs, labels in data_loader:
            print(padded_texts.shape)
            print(padded_texts_mask.shape)
            print(imgs.shape)
            print(labels.shape)
            break
