import streamlit as st
import seaborn as sns
from vqa.visualisations.vis import heatmap
import pandas as pd
import numpy as np
from vqa.utils.preprocess import split_ques
import matplotlib.pyplot as plt

class TextSelfAttMaps():

    def __init__(self, text: str, attentions: np.ndarray):

        st.markdown('### Text self-attention maps')
        st.markdown(
            'All words are not equally important to answer the question.\
            The colours here depict the importance assigned to each word by the system.\n\n'
            'However, we may end up with an incorrect answer if some key words are not attended to.\
            To prevent this, we make two predictions about the significance of words.'
        )
        self.cols = st.beta_columns(attentions.shape[0])

        self.text = split_ques(text)

        mask = np.zeros(14, dtype=np.bool)
        mask[:len(self.text)] = 1

        self.attentions = attentions[:,mask]


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
                    caption=f'Prediction #{idx}',
                    use_column_width=False,
                )

            
        
        

