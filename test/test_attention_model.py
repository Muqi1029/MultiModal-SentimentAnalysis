import pytest

import sys
sys.path.append("../")

from main import config
from models.attentionModel.model import Model
from utils import get_dataloader


class TestAttentionModel:
    @pytest.fixture(autouse=True)
    def pre_test(self):
        self.config = config
        self.config['data_path'] = "../data"
        self.config['train_path'] = "../preprocess/train.json"
        
    def test_out_dim(self):
        model = Model(self.config)
        train_data_loader = get_dataloader(self.config, 'train_path')
        for txt, txt_mask, img, label in train_data_loader:
            out = model(txt, txt_mask, img, label)
            assert out.shape == (len(txt), self.config['num_labels'])
            break