# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from vqa.utils.make_mask import make_mask
from vqa.utils.fc import FC, MLP
from vqa.utils.layer_norm import LayerNorm
from vqa.models.mcan.mca import MCA_ED

import torch.nn as nn
import torch.nn.functional as F
import torch

from collections import defaultdict

class Adapter(nn.Module):

    def __init__(self, config) -> None:

        super(Adapter, self).__init__()

        self.frcn_linear = nn.Linear(config.FRCN_FEAT_SIZE, config.HIDDEN_SIZE)

    def forward(self, image_feat: torch.Tensor) -> torch.Tensor:

        return self.frcn_linear(image_feat)

# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, config):
        super(AttFlat, self).__init__()
        self.config = config

        self.mlp = MLP(
            in_size=config.HIDDEN_SIZE,
            mid_size=config.FLAT_MLP_SIZE,
            out_size=config.FLAT_GLIMPSES,
            dropout_r=config.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            config.HIDDEN_SIZE * config.FLAT_GLIMPSES,
            config.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask, ret: defaultdict(list)):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)
        ret['att_flat'].append(att)

        att_list = []
        for i in range(self.config.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, config, token_size, answer_size):
        super(Net, self).__init__()
        self.config = config

        print(config)
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=config.WORD_EMBED_SIZE
        )


        self.lstm = nn.LSTM(
            input_size=config.WORD_EMBED_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(config)

        self.backbone = MCA_ED(config)

        # Flatten to vector
        self.attflat_img = AttFlat(config)
        self.attflat_lang = AttFlat(config)

        # Classification layers
        self.proj_norm = LayerNorm(config.FLAT_OUT_SIZE)
        self.proj = nn.Linear(config.FLAT_OUT_SIZE, answer_size)


    def forward(self, image_feat, image_feat_mask, ques_ix):
        """[summary]

        Args:
            image_feat ([type]): [description]
            image_feat_mask ([type]): [description]
            ques_ix ([type]): [description]

        Returns:
            dict: {
                "proj_feat": torch.Size(1, 3129),
                "img": {
                    "sa": [torch.Size(1, 8, 100, 100)] of 6,
                    "ca": [torch.Size(1, 8, 100, 14)] of 6,
                    "att_flat": [torch.Size([1, 100, 1])] of length 1
                },
                "text": {
                    "sa": [torch.Size([1, 8, 14, 14])] of 6,
                    "att_flat": [torch.Size([1, 14, 1])] of length 1
                }

            }
            where :
                padded_num_objects = 100
                padded_num_words = 14
                config.answer_size = 3129
                config.num_heads = 8
                config.layers=6
        """

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        img_feat = self.adapter(image_feat)

        text_ret = defaultdict(list)
        img_ret = defaultdict(list)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            image_feat_mask,
            text_ret=text_ret,
            img_ret=img_ret
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask,
            text_ret
        )

        img_feat = self.attflat_img(
            img_feat,
            image_feat_mask,
            img_ret
        )

        # Classification layers
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return_dict = {
            "proj_feat": proj_feat,
            "img": img_ret,
            "text": text_ret
        }
        return return_dict
