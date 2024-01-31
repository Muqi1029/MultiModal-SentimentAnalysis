import pytest
import sys
sys.path.append("../")
from models.baseline.model import Model
from models.imageModel import ImageModel
from main import config
from models.textModel import TextModel
from utils import get_dataloader, Trainer


class TestStrawsMan:

    @pytest.fixture(autouse=True)
    def modify_config(self):
        config['val_path'] = "../preprocess/val.json",
        config['train_path'] = "../preprocess/train.json"
        config['data_path'] = "../data/"

    def test_image_model(self):
        image_model = ImageModel(config)
        train_data_loader = get_dataloader(config, "train_path")
        for guids, txt, txt_mask, img, label in train_data_loader:
            out = image_model(img)
            assert out.size() == (len(img), config['hidden_dim']), "Failed"
            break

    def test_text_model(self):
        text_model = TextModel(config)
        train_data_loader = get_dataloader(config, "train_path")
        for txt, txt_mask, img, label in train_data_loader:
            out = text_model(txt, txt_mask)
            assert out.size() == (len(txt), config['hidden_dim']), "text model Failed!"
            break

    def test_strawMan(self):
        model = Model(config)
        train_data_loader = get_dataloader(config, "train_path")
        for txt, txt_mask, img, label in train_data_loader:
            out = model(txt, txt_mask, img, label)
            assert out.shape == (len(txt), config['num_labels'])
            break

    def test_train(self):
        train_data_loader = get_dataloader(config, "train_path")

        # 2. incorporate model
        model = Model(config)

        # 3. training
        trainer = Trainer(model, config, train_data_loader)
        # trainer.train()
    
    def test_arguments(self):
        train_data_loader = get_dataloader(config, "train_path")

        # 2. incorporate model
        model = Model(config)

        print(model.parameters())
        
