import streamlit as st
from PIL import Image
from vqa.vqa import VQA
import torch
from vqa.visualisations.vis import show_architecture
from vqa.visualisations.vis import hbarplot
import pandas as pd
from slit.bb import BoundingBox
from slit.attmaps import TextSelfAttMaps
import numpy as np
from sidebar import SideBar

st.title('AMA: Visualizing attentions for Visual Question Answering')

sb = SideBar()

model_name = sb.model_name
question = sb.question
image_idx = sb.image_idx

st.markdown("### Model Architecture")
show_architecture(model_name)

# Load the VQA model just after UI is loaded
if (model_name is not None):
    vqa_object = VQA(model_name)

image = None
image_feat = None
bboxes = None
if (image_idx is not None):
    image = np.array(Image.open(f'assets/images/{image_idx}.jpg').convert('RGB'))
    feats = np.load(f'assets/feats/{image_idx}.npz')

    image_feat = torch.tensor(feats['x'].T) #(num_objects, 2048)
    bboxes = torch.tensor(feats['bbox']) #(num_objects, 4)

# Call this only when question and image have loaded
if (question is not None and image is not None):

    # Get the dict from the net
    ret = vqa_object.inference(question, image_feat)

    st.markdown('### Predicted confidence of top-7 answers')
    vqa_object.answer_confidence_plot(ret)

    bboxes = BoundingBox.get_top_bboxes(ret['img']['iatt_maps'].squeeze().transpose(1,0), bboxes)

    bb_obj = BoundingBox(image, bboxes=bboxes)

    # question is the question string, and att is a nd.ndarray of shape (n_glimpses, num_words)
    TextSelfAttMaps(question, attentions=ret['text']['qatt'].squeeze().transpose(1,0).detach().numpy())