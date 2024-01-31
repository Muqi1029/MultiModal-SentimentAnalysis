from torchvision.transforms import Compose, ToTensor, Resize
from transformers import BertTokenizer
import torch
from torch import optim
from functools import partial
from tqdm import tqdm
from transformers import BertTokenizer
from functools import partial
from main import config
from dataset import MultiModalDataset
from torch.utils.data import DataLoader
from torchmultimodal.models.flava.model import flava_model_for_classification

import json


tokenizer = BertTokenizer.from_pretrained(
    "/root/.cache/huggingface/hub/models--bert-base-uncased", padding="max_length", max_length=64)

def flava_transform(tokenizer, input):
    """      return guid, img, text, label """
    batch = {}
    image_transform = Compose([ToTensor(), Resize([224,224])])
    image = torch.stack([image_transform(d[1].convert("RGB")) for d in input])
    batch["image"] = image

    texts = [d[2] for d in input]
    tokenized=tokenizer(texts,return_tensors='pt',padding="max_length",max_length=64)
    batch.update(tokenized)

    batch["answers"] = torch.tensor([d[3] for d in input], dtype=torch.long)

    return batch

transform=partial(flava_transform, tokenizer)

def get_dataloader(config, path_key, transform):
    data = json.load(open(config[path_key], "r", encoding="utf-8"))
    guid = [d['guid'] for d in data]  # 3200
    labels = [d['label'] for d in data]
    multiModalDataset = MultiModalDataset(imgs_dir=config['data_path'],
                                          text_dir=config['data_path'],
                                          labels=labels, guids=guid, config=config)

    train_data_loader = DataLoader(multiModalDataset, batch_size=config['batch_size'],
                                   collate_fn=transform)
    return train_data_loader


model = flava_model_for_classification(config['num_labels'])
optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
train_data_loader = get_dataloader(config, "train_path", transform=transform)
val_data_loader = get_dataloader(config, "val_path", transform=transform)
test_data_loader = get_dataloader(config, "test_path", transform=transform)

for epoch in tqdm(range(config['num_epochs'])):

    for idx, batch in enumerate(train_data_loader):
        optimizer.zero_grad()
        out = model(text=batch['input_ids'], image=batch['image'], labels=batch['answers'])
        loss = out.loss
        
        predictions = out.logits.argmax(dim=-1)
        acc = torch.eq(predictions, batch['answers']).float().mean()
        print(f"acc: {acc:.2%}")
        
        loss.backward()
        optimizer.step()
        print(f"Loss at step {idx} = {loss}")
    
    # eval
    model.eval()
    print("Start to eval:".center("-", 50))
    with torch.no_grad():
        acc = 0
        for idx, batch in enumerate(val_data_loader):
            out = model(text=batch['input_ids'], image=batch['image'], labels=batch['answers'])
            acc += torch.eq(out.logits.argmax(dim=-1), batch['answers']).float()
        acc /= len(val_data_loader.dataset)
    print(f"eval result: {acc:.2f}".center("-", 50))
    
    
# predict
results = []
for batch in test_data_loader:
    out = model(text=batch['input_ids'], image=batch['image'], labels=batch['answers'])
    results.extend(out.logits.argmax(dim=-1).to_array())
