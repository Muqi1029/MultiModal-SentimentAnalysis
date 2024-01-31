import json
from typing import Dict
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Compose, Resize, ToTensor
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import MultiModalDataset
from transformers import AutoTokenizer
from functools import partial
import os
import matplotlib.pylab as plt


def collate_fn(config, batch):
    """      return guid, img, text, label """
    guids = [d[0] for d in batch]

    img_transform = Compose([
        Resize((224, 224)), ToTensor()]
    )
    imgs = torch.stack([img_transform(d[1]) for d in batch]).float()

    tokenizer = AutoTokenizer.from_pretrained(config['bert_name'])

    tokenized_texts = [torch.LongTensor(
        tokenizer.convert_tokens_to_ids(tokenizer(d[2]))) for d in batch]

    labels = torch.tensor([d[3] for d in batch], dtype=torch.long)

    texts_mask = [torch.ones_like(text) for text in tokenized_texts]

    padded_texts = pad_sequence(
        tokenized_texts, batch_first=True, padding_value=0)
    padded_texts_mask = pad_sequence(
        texts_mask, batch_first=True, padding_value=0).gt(0)

    return guids, padded_texts, padded_texts_mask, imgs, labels


def get_dataloader(config, path_key):
    data = json.load(open(config[path_key], "r", encoding="utf-8"))
    guid = [d['guid'] for d in data]  # 3200
    labels = [d['label'] for d in data]
    multiModalDataset = MultiModalDataset(imgs_dir=config['data_path'],
                                          text_dir=config['data_path'],
                                          labels=labels, guids=guid, config=config)

    train_data_loader = DataLoader(multiModalDataset, batch_size=config['batch_size'],
                                   collate_fn=partial(collate_fn, config))
    return train_data_loader


class Trainer:
    def __init__(self, model: nn.Module, config: Dict, train_data_loader: DataLoader = None, val_data_loader: DataLoader = None,
                 test_data_loader_loader: DataLoader = None):
        config['optim'] = optim.AdamW(
            model.parameters(), lr=config['learning_rate'])

        self.model = model
        self.config = config
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader_loader

    def train(self):
        val_acc_history = []
        best_acc = 0
        for epoch in tqdm(range(self.config['num_epochs'])):
            for i, (guids, padded_texts, padded_texts_mask, imgs, labels) in enumerate(self.train_data_loader):
                self.model.train()

                padded_texts, padded_texts_mask, imgs, labels = padded_texts.to(self.config['device']), \
                    padded_texts_mask.to(self.config['device']), \
                    imgs.to(self.config['device']), \
                    labels.to(self.config['device'])

                y_pred = self.model(
                    padded_texts, padded_texts_mask, imgs, labels)

                # compute loss
                loss = self.config['loss_fn'](y_pred, labels)

                with torch.no_grad():
                    acc = torch.eq(labels, y_pred.argmax(dim=-1)).float().mean()
                    
                # optimize
                self.config['optim'].zero_grad()
                loss.backward()
                self.config['optim'].step()
                print(
                    f"Epoch [{epoch+1}/{self.config['num_epochs']}] Iteration [{i + 1}/{len(self.train_data_loader)}] loss: {loss.item():.4f} acc: {acc:.2%}")

            # evaluation on the validation dataset
            print()
            print(
                f"Epoch {epoch+1}: start to eval on val dataset".center(50, '='))
            acc = self.eval(self.val_data_loader)
            print(f"Eval over: val acc = {acc:.2%}".center(50, '='))
            print()
            
            val_acc_history.append(acc)
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), self.config['model_path'])
                print(f"store model in {self.config['model_path']}")
        plt.plot(list(range(1, len(val_acc_history) + 1)), val_acc_history)
        plt.xlabel("epoch")
        plt.ylabel("val accuracy")
        plt.title("Training process")
        plt.savefig("./output/training.png", dpi=800)

    def eval(self, data_loader: DataLoader):
        self.model.eval()
        num = 0
        with torch.no_grad():
            for guids, padded_texts, padded_texts_mask, imgs, labels in tqdm(data_loader):
                padded_texts, padded_texts_mask, imgs, labels = padded_texts.to(self.config['device']), \
                    padded_texts_mask.to(self.config['device']), \
                    imgs.to(self.config['device']), \
                    labels.to(self.config['device'])

                pred_labels = self.model(
                    padded_texts, padded_texts_mask, imgs).squeeze()
                pred_labels = torch.argmax(pred_labels, dim=1)
                # print(f"{torch.eq(pred_labels, labels).sum().item()}")
                num += torch.eq(pred_labels, labels).sum().item()
        acc = num / len(data_loader.dataset)
        return acc

    def predict(self, test_data_loader: DataLoader, file_path):
        results = []
        self.model.eval()

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with torch.no_grad():
            for guids, padded_texts, padded_texts_mask, imgs, labels in tqdm(test_data_loader):
                padded_texts, padded_texts_mask, imgs, labels = padded_texts.to(self.config['device']), \
                    padded_texts_mask.to(self.config['device']), \
                    imgs.to(self.config['device']), \
                    labels.to(self.config['device'])
                pred_probs = self.model(
                    padded_texts, padded_texts_mask, imgs, labels)
                pred_labels = torch.argmax(pred_probs, dim=1)
                for guid, label in zip(guids, pred_labels):
                    results.append((guid, label.item()))

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("guid,tag\n")
            for guid, tag in results:
                if tag == 0:
                    tag = "negative"
                elif tag == 1:
                    tag = "neutral"
                else:
                    tag = "positive"
                f.write(f"{guid},{tag}\n")
