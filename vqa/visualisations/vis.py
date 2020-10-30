import seaborn as sns
import torch
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import os
from PIL import Image, ImageDraw

def ax2im(ax):
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches = 'tight', pad_inches = 0)
    buf.seek(0)
    im = Image.open(buf)
    return im

def hbarplot(x, y, data, **kwargs):

    plt.clf()
    ax = sns.barplot(x=x, y=y, data=data, orient='h', palette="Blues_d", **kwargs)
    im = ax2im(ax)
    
    return im

def heatmap(data, annot, **kwargs):

    ax = sns.heatmap(data=data, annot=annot, fmt="s", cmap='rocket', cbar=True, square=True, xticklabels=False, yticklabels=False, **kwargs)
    im = ax2im(ax)
    
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