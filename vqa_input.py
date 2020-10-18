import streamlit as st
from PIL import Image
from vqa.vqa import VQA
import torchvision
import torch

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
if (image_uploaded is not None):
    image = Image.open(image_uploaded)

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
if (question and image):

    # frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91, pretrained_backbone=True)
    # frcnn.backbone.fpn = None

    # frcnn.eval()

    # image_feat = torchvision.transforms.functional.to_tensor(image)

    vqa_object.inference(question, image)


