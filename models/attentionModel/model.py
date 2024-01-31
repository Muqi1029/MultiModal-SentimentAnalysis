from models.imageModel import ImageModel
from models.textModel import TextModel
from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_model = ImageModel(config)
        self.text_model = TextModel(config)

        self.attention = nn.TransformerEncoderLayer(
            d_model=config['hidden_dim'] * 2,
            nhead=config['num_heads'],
            dropout=config['attention_dropout']
        )

        self.linear = nn.Sequential(
            nn.Dropout(config['fusion_dropout']),
            nn.Linear(2 * config['hidden_dim'], config['hidden_dim']),
            nn.ReLU(inplace=True),
            nn.Dropout(config['fusion_dropout']),
            nn.Linear(config['hidden_dim'], config['num_labels'])
        )

    def forward(self, txt=None, txt_mask=None, img=None, labels=None):
        assert not self.config['only_txt'] or not self.config['only_img'], "Cannot only have either txt or img"

        if self.config["only_txt"]:
            txt_feature = self.text_model(txt, txt_mask)
            return self.linear(self.attention(torch.cat([txt_feature.unsqueeze(0), torch.zeros_like(txt_feature).unsqueeze(0)], dim=2).squeeze()))
        if self.config["only_img"]:
            img_feature = self.image_model(img)
            return self.linear(self.attention(torch.cat([torch.zeros_like(img_feature).unsqueeze(0), img_feature.unsqueeze(0)], dim=2)).squeeze())
        
        txt_feature = self.text_model(txt, txt_mask)
        img_feature = self.image_model(img)
        attention_out = self.attention(torch.cat(
            [txt_feature.unsqueeze(0), img_feature.unsqueeze(0)], dim=2)).squeeze()

        return self.linear(attention_out)
