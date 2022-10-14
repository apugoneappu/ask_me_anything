import pickle
import yaml
from dotmap import DotMap
from vqa.models.mfb.net import Net as Net_MFB
from vqa.models.mcan.net import Net as Net_MCAN
from vqa.utils.preprocess import proc_ques
import torch
from vqa.utils.make_mask import make_mask
from vqa.visualisations.vis import hbarplot
import streamlit as st
import pandas as pd
import os
import gdown

@st.cache
class VQA():

    def __init__(self, model_name):
        """
        Args:
            model_name (str): The corrected model string ('mfb', 'mcan', etc)
        """

        # such as 'mfb', 'mcan' etc.
        self.model_name = model_name

        with open(f'vqa/configs/{self.model_name}.yml', 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            self.config = DotMap(self.config)

        self.token_size = 20573
        self.answer_size = 3129

        self.net = self.get_net(self.model_name)

        with open('vqa/pickles/dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
        
        self.token_to_ix = dataset['token_to_ix']
        self.ix_to_answer = dataset['ix_to_ans']


    def get_net(self, model_name):

        url = 'https://drive.google.com/uc?id='
        output = 'vqa/pickles/'

        if (model_name == 'mfb'):

            net = Net_MFB(self.config, token_size=self.token_size, answer_size=self.answer_size)

            url += '1uShGcr6V_kDKN1mBgu35CiRUDahEiobD'
            output += 'mfb.pkl'
        
        # elif (model_name == 'mcan'):

        #     net = Net_MCAN(self.config, token_size=self.token_size, answer_size=self.answer_size)
            
        #     url += '1nZGePRdZ-tYJlK2GT-V9nA83YhlGDb7w'
        #     output += 'mcan.pkl'

        else:
            raise NotImplementedError
        
        net.eval()
        if (not os.path.isfile(f'./vqa/pickles/{model_name}.pkl')):
            gdown.download(url, output, quiet=False)

        net.load_state_dict(torch.load(f'./vqa/pickles/{model_name}.pkl', map_location='cpu')['state_dict'], strict=False)
        
        return net
    
    def inference(self, question, image_feat):
        """Function used for running inverence given the image and question

        Args:
            question (str): The question
            image (np.ndarray): The image

        Returns:
            [dict]: {
                "img": {
                    "iatt_maps": torch.Tensor(1, 100, 2)
                },
                "text" {
                    "qatt": torch.Tensor(1, 14, 2)
                }
            }
        """

        # extract indices of question words
        ques_ix = proc_ques(question, self.token_to_ix, max_token=14)
        ques_ix = torch.tensor(ques_ix).unsqueeze(0)

        # pad image features
        frcn_feat = torch.zeros(100, 2048)
        frcn_feat[:image_feat.shape[0]] = image_feat
        image_feat = frcn_feat.unsqueeze(0)

        image_feat_mask = make_mask(image_feat)

        ret = self.net.forward(image_feat, image_feat_mask, ques_ix)

        return ret

    
    def answer_confidence_plot(self, ret, k=7):

        soft_proj = torch.softmax(ret['proj_feat'], dim=-1)
        values, indices = torch.topk(soft_proj, k)

        values, indices = values.squeeze(0), indices.squeeze(0)

        df = {}
        df['answers'] = []
        df['confidence'] = []
        for idx in range(indices.shape[0]):
            df['answers'].append(self.ix_to_answer[str(indices[idx].item())])
            df['confidence'].append(100*values[idx].item())

        df = pd.DataFrame(df)

        # y axis should have categorical variable
        st.image(hbarplot(y='answers', x='confidence', data=df))
