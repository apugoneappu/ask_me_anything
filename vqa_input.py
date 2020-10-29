import streamlit as st
from PIL import Image
from vqa.vqa import VQA
import torchvision
import torch
from vqa.visualisations.vis import show_architecture
from vqa.visualisations.vis import hbarplot
import pandas as pd
from slit.bb import BoundingBox
import numpy as np

st.title('Visualising attentions for Visual Question Answering (VQA) systems')

model_name = st.selectbox(
    label = 'Please choose the model for which you want to see attentions',
    options = [
        'MFB: Multi-modal Factorized Bilinear Pooling with Co-Attention Learning',
        'MCAN: Deep Modular Co-Attention Networks'
    ],
    index = 0,
    key = 'model_name'
)

show_architecture(model_name)

question = st.text_input(
    label = 'Please type your question here',
    value= 'What is in the image?',  
    key= 'question'
)

image_uploaded = st.file_uploader(
    label = "Please upload the image here", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=False, 
    key='image_upload'
)

# Load the VQA model just after UI is loaded
if (model_name is not None):
    vqa_object = VQA(model_name)

image = None
# image_uploaded = open('/Users/apoorve/Downloads/dont_worry.jpg', 'rb')
if (image_uploaded is not None):
    image = np.array(Image.open(image_uploaded).convert('RGB'))

    st.image(
        image, 
        caption='Uploaded image', 
        width=None, 
        use_column_width=True, 
        clamp=False, 
        channels='RGB',
        output_format='auto', 
    )

# Call this only when question and image have loaded
if (question and image is not None):

    # Get the dict from the net
    ret = vqa_object.inference(question, image)

    st.markdown('### Predicted confidence of top-7 answers')
    vqa_object.answer_confidence_plot(ret)

    st.markdown('### Plotting top-attended bounding boxes')

    bb_obj = BoundingBox(
        image, 
        bboxes=[
            # [xmin, xmax, ymin, ymax, confidence]
            [0.1, 0.3, 0.2, 0.6, 0.9],
            [0.2, 0.5, 0.7, 0.9, 0.7],
            [0.7, 0.85, 0.01, 0.15, 0.02],
            [0.5, 0.65, 0.8, 0.9, 0.09],
            [0.0, 0.2, 0.8, 1.0, 0.03],
            [0.0, 0.5, 0.0, 0.1, 0.33]
        ]
    )






