import seaborn
import torch
import io
from PIL import Image
import matplotlib.pyplot as plt

def hbarplot(x, y, data):

    ax = seaborn.barplot(x=x, y=y, data=data, orient='h')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    return im
