from torch import nn
import torch
from ..imageModel import ImageModel
from ..textModel import TextModel
from enum import Enum


class NAIVE_CONN(Enum):
    ADD_CONN = 1
    CAT_CONN = 2


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_model = ImageModel(config)
        self.text_model = TextModel(config)

        self.naive_conn = NAIVE_CONN.ADD_CONN if config['conn_way'] == 1 else NAIVE_CONN.CAT_CONN

        if self.naive_conn == NAIVE_CONN.CAT_CONN:
            self.linear_net = nn.Linear(config['hidden_dim'] * 2, config['hidden_dim'])
        elif self.naive_conn == NAIVE_CONN.ADD_CONN:
            self.linear_net = nn.Linear(config['hidden_dim'], config['hidden_dim'])
        
        self.out_linear = nn.Linear(config['hidden_dim'], config['num_labels'])

    def forward(self, txt=None, txt_mask=None, img=None, labels=None):
        assert not self.config['only_txt'] or not self.config['only_img'], "at least input either of img or txt"
        if self.config['only_txt'] == True:
            txt_feature = self.text_model(txt, txt_mask)
            return self.out_linear(txt_feature)
            
        if self.config['only_img'] == True:
            img_feature = self.image_model(img)
            return self.out_linear(img_feature)
            
        if self.config['only_img'] == False and self.config['only_txt'] == False:
            img_feature = self.image_model(img)
            txt_feature = self.text_model(txt, txt_mask)
            conn_feature = None
            if self.naive_conn == NAIVE_CONN.CAT_CONN:
                conn_feature = torch.cat([img_feature, txt_feature], dim=1)
            elif self.naive_conn == NAIVE_CONN.ADD_CONN:
                conn_feature = torch.add(img_feature, txt_feature)
            out = self.out_linear(self.linear_net(conn_feature))
            return out


