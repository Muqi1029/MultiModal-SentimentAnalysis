import argparse
import torch
from torch import nn
from utils import Trainer, get_dataloader
from models.baseline.model import Model as BaselineModel
from models.attentionModel.model import Model as AttentionModel

config = {
    "learning_rate": 0.005,
    "loss_fn": nn.CrossEntropyLoss(),
    "num_epochs": 8,
    "train_path": "./preprocess/train.json",
    "val_path": "./preprocess/val.json",
    "test_path": "./preprocess/test.json",
    "data_path": "./data/",
    "batch_size": 32,
    "hidden_dim": 64,
    "num_labels": 3,
    "finetune": False,
    "bert_name": '/root/.cache/huggingface/hub/models--roberta-base', # The path to your pretrained LM
    "bert_dropout": 0.5,
    "img_dropout": 0.5,
    "attention_dropout": 0.5,
    "fusion_dropout": 0.5,
    "conn_way": 2,
    "num_heads": 8,
    "device": None,
    "model_type": "attention" ,
    "only_img": False,
    "only_txt": False,
    "model_path": "./model.bin"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conn_way", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)

    parser.add_argument("--only_txt", action="store_true")
    parser.add_argument("--only_img", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--model_type", type=str, default="baseline", choices=['baseline', 'attention', 'flava'])
    
    args = parser.parse_args()
    
    config['hidden_dim'] = args.hidden_dim
    config['only_img'] = args.only_img
    config['only_txt'] = args.only_txt
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['conn_way'] = args.conn_way
    config['model_type'] = args.model_type

    if args.train:
        config['mode'] = "train"
    elif args.predict:
        config['mode'] = "predict"


config['device'] = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    # 1. config dataset and to get train_data_loader
    train_data_loader = get_dataloader(config, "train_path")
    val_data_loader = get_dataloader(config, "val_path")
    
    # 2. incorporate model
    model = None
    if config['model_type'] == "baseline":
        model = BaselineModel(config)
        print("use baseline model")
    elif config['model_type'] == "attention":
        model = AttentionModel(config)
        print("use attention model")
    
    model.to(config['device'])
    print(f"Move model to {config['device']}")

    # 3. training
    trainer = Trainer(model, config, train_data_loader, val_data_loader, None)
    trainer.train()

def predit():
    if config['model_type'] == "baseline":
        model = BaselineModel(config)
        print("use baseline model")

    elif config['model_type'] == "attention":
        model = AttentionModel(config)
        print("use attention model")
        
    model.load_state_dict(torch.load(config['model_path']))
    
    test_data_loader = get_dataloader(config, "test_path")
    trainer = Trainer(model, config)
    model.to(config['device'])
    print(f"move model to {config['device']}")

    print("starting to predict: ")
    trainer.predict(test_data_loader=test_data_loader, file_path="./output/test.txt")


if __name__ == '__main__':
    parse_args()
    if config['mode'] == "train":
        main()
    elif config['mode'] == "predict":
        predit()
    # predit()
