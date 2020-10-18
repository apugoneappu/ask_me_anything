import seaborn as sns
import torch
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import os
from PIL import Image, ImageDraw

def hbarplot(x, y, data):

    ax = sns.barplot(x=x, y=y, data=data, orient='h')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    return im

def show_architecture(model_name):

    if ('MFB' in model_name):
        st.image('vqa/visualisations/mfb.png', 
        caption='MFB architecture',
        use_column_width=True
    )
    elif ('MCAN' in model_name):
        st.image('vqa/visualisations/mcan.png', 
        caption='MCAN architecture',
        use_column_width=True
    )