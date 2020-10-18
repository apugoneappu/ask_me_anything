# --------------------------------------------------------
# OpenVQA
# Licensed under The MIT License [see LICENSE for details]
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

from vqa.utils.fc import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# ---- Multi-Model Hign-order Bilinear Pooling Co-Attention----
# -------------------------------------------------------------


class MFB(nn.Module):
    def __init__(self, config, img_feat_size, ques_feat_size, is_first):
        super(MFB, self).__init__()
        self.config = config
        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, config.MFB_K * config.MFB_O)
        self.proj_q = nn.Linear(ques_feat_size, config.MFB_K * config.MFB_O)
        self.dropout = nn.Dropout(config.DROPOUT_R)
        self.pool = nn.AvgPool1d(config.MFB_K, stride=config.MFB_K)

    def forward(self, img_feat, ques_feat, exp_in=1, **kwargs):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)
        exp_out = img_feat * ques_feat                  # (N, C, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)     # (N, C, K*O)
        z = self.pool(exp_out) * self.config.MFB_K         # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))         # (N, C*O)
        z = z.view(batch_size, -1, self.config.MFB_O)      # (N, C, O)
        return z, exp_out


class QAtt(nn.Module):
    def __init__(self, config):
        super(QAtt, self).__init__()
        self.config = config
        self.mlp = MLP(
            in_size=config.LSTM_OUT_SIZE,
            mid_size=config.HIDDEN_SIZE,
            out_size=config.Q_GLIMPSES,
            dropout_r=config.DROPOUT_R,
            use_relu=True
        )

    def forward(self, ques_feat, ques_feat_mask, text_ret):
        '''
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            qatt_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
        '''
        qatt_maps = self.mlp(ques_feat)                 # (N, T, Q_GLIMPSES)
        if ques_feat_mask is not None:
            qatt_maps = qatt_maps.masked_fill(ques_feat_mask, -1e9)

        qatt_maps = F.softmax(qatt_maps, dim=1)         # (N, T, Q_GLIMPSES)
        assert int(round(torch.sum(qatt_maps).item())) == qatt_maps.shape[0] * qatt_maps.shape[2]
        text_ret["qatt"] = qatt_maps

        qatt_feat_list = []
        for i in range(self.config.Q_GLIMPSES):
            mask = qatt_maps[:, :, i:i + 1]             # (N, T, 1)
            mask = mask * ques_feat                     # (N, T, LSTM_OUT_SIZE)
            mask = torch.sum(mask, dim=1)               # (N, LSTM_OUT_SIZE)
            qatt_feat_list.append(mask)
        qatt_feat = torch.cat(qatt_feat_list, dim=1)    # (N, LSTM_OUT_SIZE*Q_GLIMPSES)

        return qatt_feat


class IAtt(nn.Module):
    def __init__(self, config, img_feat_size, ques_att_feat_size):
        super(IAtt, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(config.DROPOUT_R)
        self.mfb = MFB(config, img_feat_size, ques_att_feat_size, True)
        self.mlp = MLP(
            in_size=config.MFB_O,
            mid_size=config.HIDDEN_SIZE,
            out_size=config.I_GLIMPSES,
            dropout_r=config.DROPOUT_R,
            use_relu=True
        )

    def forward(self, img_feat, ques_att_feat, img_feat_mask, img_ret):
        '''
            img_feats.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_att_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
            iatt_feat.size() -> (N, MFB_O * I_GLIMPSES)
        '''
        ques_att_feat = ques_att_feat.unsqueeze(1)      # (N, 1, LSTM_OUT_SIZE * Q_GLIMPSES)
        img_feat = self.dropout(img_feat)
        z, _ = self.mfb(img_feat, ques_att_feat)        # (N, C, O)

        iatt_maps = self.mlp(z)                         # (N, C, I_GLIMPSES)
        if img_feat_mask is not None:
            iatt_maps = iatt_maps.masked_fill(img_feat_mask, -1e9)

        iatt_maps = F.softmax(iatt_maps, dim=1)         # (N, C, I_GLIMPSES)
        assert int(round(torch.sum(iatt_maps).item())) == iatt_maps.shape[0] * iatt_maps.shape[2]
        img_ret["iatt_maps"] = iatt_maps
        
        iatt_feat_list = []
        for i in range(self.config.I_GLIMPSES):
            mask = iatt_maps[:, :, i:i + 1]             # (N, C, 1)
            mask = mask * img_feat                      # (N, C, FRCN_FEAT_SIZE)
            img_ret["iatt_feat"] = mask.clone()
            mask = torch.sum(mask, dim=1)               # (N, FRCN_FEAT_SIZE)
            iatt_feat_list.append(mask)
        iatt_feat = torch.cat(iatt_feat_list, dim=1)    # (N, FRCN_FEAT_SIZE*I_GLIMPSES)
        return iatt_feat


class CoAtt(nn.Module):
    def __init__(self, config):
        super(CoAtt, self).__init__()
        self.config = config

        img_feat_size = config.FRCN_FEAT_SIZE
        img_att_feat_size = img_feat_size * config.I_GLIMPSES
        ques_att_feat_size = config.LSTM_OUT_SIZE * config.Q_GLIMPSES

        self.q_att = QAtt(config)
        self.i_att = IAtt(config, img_feat_size, ques_att_feat_size)

        if self.config.HIGH_ORDER:  # MFH
            self.mfh1 = MFB(config, img_att_feat_size, ques_att_feat_size, True)
            self.mfh2 = MFB(config, img_att_feat_size, ques_att_feat_size, False)
        else:  # MFB
            self.mfb = MFB(config, img_att_feat_size, ques_att_feat_size, True)

    def forward(self, img_feat, ques_feat, img_feat_mask, ques_feat_mask, text_ret, img_ret, **kwargs):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        ques_feat = self.q_att(ques_feat, ques_feat_mask, text_ret)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_feat = self.i_att(img_feat, ques_feat, img_feat_mask, img_ret)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)

        if self.config.HIGH_ORDER:  # MFH
            z1, exp1 = self.mfh1(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))        # z1:(N, 1, O)  exp1:(N, C, K*O)
            z2, _ = self.mfh2(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1), exp1)     # z2:(N, 1, O)  _:(N, C, K*O)
            z = torch.cat((z1.squeeze(1), z2.squeeze(1)), 1)                            # (N, 2*O)
        else:  # MFB
            z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
            z = z.squeeze(1)                                                            # (N, O)

        return z
