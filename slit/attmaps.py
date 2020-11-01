import streamlit as st
import seaborn as sns
from vqa.visualisations.vis import heatmap
import pandas as pd
import numpy as np
from vqa.utils.preprocess import split_ques
import matplotlib.pyplot as plt

class TextSelfAttMaps():

    def __init__(self, text: str, attentions: np.ndarray):

        self.text = split_ques(text)

        mask = np.zeros(14, dtype=np.bool)
        mask[:len(self.text)] = 1

        self.attentions = attentions[:,mask]

        st.markdown('### Text self-attention maps')
        self.cols = st.beta_columns(attentions.shape[0])

        self.plot_maps()

    def plot_maps(self):
        """
        attentions = np.ndarray(n_heads, num_words)
        """


        for idx in range(self.attentions.shape[0]):

            data = pd.DataFrame(
                {"attentions": self.attentions[idx]},
                index = self.text
            )

            annot = np.array(self.text).reshape(data.shape)

            plt.clf()
            with self.cols[idx]:
                im = heatmap(data, annot)
                st.image(
                    im,
                    caption=f'Glimpse #{idx}',
                    use_column_width=False,
                )

            
        
        

