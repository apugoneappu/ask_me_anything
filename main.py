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

st.title('Visualizing attentions for Visual Question Answering (VQA)')

sb = SideBar()

model_name = sb.model_name
question = sb.question
image_idx = sb.image_idx

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

image = None
if (image_uploaded is None):
    image_uploaded = open('assets/test.jpg', 'rb')

if (image_uploaded is not None):
    image = np.array(Image.open(image_uploaded).convert('RGB'))

    st.sidebar.image(
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


    bb_obj = BoundingBox(
        image, 
        bboxes=[
            # [xmin, xmax, ymin, ymax, confidence]
            # examples
            [0.1, 0.3, 0.2, 0.6, 0.9],
            [0.2, 0.5, 0.7, 0.9, 0.7],
            [0.7, 0.85, 0.01, 0.15, 0.02],
            [0.5, 0.65, 0.8, 0.9, 0.09],
            [0.0, 0.2, 0.8, 1.0, 0.03],
            [0.0, 0.5, 0.0, 0.1, 0.33]
        ]
    )

    # question is the question string, and att is a nd.ndarray of shape (n_glimpses, num_words)
    TextSelfAttMaps(question, attentions=ret['text']['qatt'].squeeze().transpose(1,0).detach().numpy())