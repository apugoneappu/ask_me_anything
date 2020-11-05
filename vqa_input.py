import streamlit as st
from PIL import Image
from vqa.vqa import VQA
import torchvision
import cv2
import torch
from vqa.visualisations.vis import show_architecture
from vqa.visualisations.vis import hbarplot
import pandas as pd
from slit.bb import BoundingBox
from slit.attmaps import TextSelfAttMaps
import numpy as np
from demo_feature_extraction import FrcnFeatures

st.title('Visualizing attentions for Visual Question Answering (VQA)')

st.sidebar.title('Inputs')

model_name = st.sidebar.selectbox(
    label = 'Please choose the model for which you want to see attentions',
    options = [
        'MFB: Multi-modal Factorized Bilinear Pooling with Co-Attention Learning',
        'MCAN: Deep Modular Co-Attention Networks'
    ],
    index = 0,
    key = 'model_name'
)

st.markdown("### Model Architecture")
show_architecture(model_name)

question = st.sidebar.text_input(
    label = 'Please type your question here',
    value= 'What is there in the image?',  
    key= 'question'
)

image_uploaded = st.sidebar.file_uploader(
    label = "Please upload the image here", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=False, 
    key='image_upload'
)

# Load the VQA model just after UI is loaded
if (model_name is not None):
    vqa_object = VQA(model_name)

frcnn = FrcnFeatures()
image = None
if (image_uploaded is None):
    image_uploaded = open('assets/test.jpg', 'rb')

if (image_uploaded is not None):
    image = np.array(Image.open(image_uploaded).convert('RGB').resize((224,224)))

    st.sidebar.image(
        image, 
        caption='Uploaded image', 
        width=None, 
        use_column_width=True, 
        clamp=False, 
        channels='RGB',
        output_format='auto', 
    )

    bboxes, image_feat = frcnn(image)

# Call this only when question and image have loaded
if (question and image is not None):

    import ipdb; ipdb.set_trace()
    # Get the dict from the net
    ret = vqa_object.inference(question, image_feat)

    st.markdown('### Predicted confidence of top-7 answers')
    vqa_object.answer_confidence_plot(ret)


    bboxes = torch.load('/Users/apoorve/Desktop/Personal/streamlit_share_demo.nosync/vqa/pickles/bbox.pt')
    iatt_maps = ret['img']['iatt_maps'].squeeze().transpose(1,0).detach().numpy() #shape (2, 100)
    
    bboxes_sorted = BoundingBox.get_top_bboxes(iatt_maps, bboxes)

    # bboxes should be shape(num_glimpses, k, 5)
    bb_obj = BoundingBox(image, bboxes= bboxes_sorted)

    # question is the question string, and att is a nd.ndarray of shape (n_glimpses, num_words)
    TextSelfAttMaps(question, attentions=ret['text']['qatt'].squeeze().transpose(1,0).detach().numpy())