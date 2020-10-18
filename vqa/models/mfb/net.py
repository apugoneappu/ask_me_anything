# --------------------------------------------------------
# OpenVQA
# Licensed under The MIT License [see LICENSE for details]
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------
from vqa.utils.make_mask import make_mask
from vqa.models.mfb.mfb import CoAtt
import torch
import torch.nn as nn

from collections import defaultdict

# -------------------------------------------------------
# ---- Main MFB/MFH model with Co-Attention Learning ----
# -------------------------------------------------------


class Net(nn.Module):
    def __init__(self, config, token_size, answer_size):
        super(Net, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=config.WORD_EMBED_SIZE
        )

        self.lstm = nn.LSTM(
            input_size=config.WORD_EMBED_SIZE,
            hidden_size=config.LSTM_OUT_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(config.DROPOUT_R)
        self.dropout_lstm = nn.Dropout(config.DROPOUT_R)
        self.backbone = CoAtt(config)

        if config.HIGH_ORDER:      # MFH
            self.proj = nn.Linear(2*config.MFB_O, answer_size)
        else:                   # MFB
            self.proj = nn.Linear(config.MFB_O, answer_size)

    def forward(self, img_feat, img_feat_mask, ques_ix, **kwargs):

        # Pre-process Language Feature
        ques_feat_mask = make_mask(ques_ix.unsqueeze(2)).squeeze(2).permute(0, 2, 1)
        ques_feat = self.embedding(ques_ix)     # (N, T, WORD_EMBED_SIZE)
        ques_feat = self.dropout(ques_feat)
        ques_feat, _ = self.lstm(ques_feat)     # (N, T, LSTM_OUT_SIZE)
        ques_feat = self.dropout_lstm(ques_feat)

        text_ret = defaultdict(list)
        img_ret = defaultdict(list)

        z = self.backbone(
            img_feat,
            ques_feat,
            img_feat_mask.squeeze(2).permute(0, 2, 1),
            ques_feat_mask,
            text_ret=text_ret,
            img_ret=img_ret
        )  # MFH:(N, 2*O) / MFB:(N, O)
        proj_feat = self.proj(z)                # (N, answer_size)

        return_dict = {
            "proj_feat": proj_feat,
            "img": img_ret,
            "text": text_ret
        }
        return return_dict