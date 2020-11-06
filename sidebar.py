import streamlit as st
import random

class SideBar():

    def __init__(self, num_images=32) -> None:

        self.num_images = num_images

        self.title = "Ask me anything (AMA)"
        self.model_name = None
        self.question = None
        self.image_idx = None

        self._title()
        self._model()
        self._question()
        self._image()

        self._show_images()
    
    def _title(self):
        st.sidebar.title(self.title)

    def _model(self):

        st.sidebar.markdown('## Step 1: Choose model')

        self.model_name = st.sidebar.selectbox(
            label = 'Please choose the model here',
            options = [
                'MFB: Multi-modal Factorized Bilinear Pooling with Co-Attention Learning',
                # 'MCAN: Deep Modular Co-Attention Networks'
            ],
            index = 0,
            key = 'model_name'
        )

        self._fix_model_name(self.model_name)
    
    def _question(self):
        st.sidebar.markdown('## Step 2: Enter question')

        self.question = st.sidebar.text_input(
            label = 'Please type your question here',
            value= 'What is there in the image?',  
            key= 'question'
        )

    
    def _image(self):

        st.sidebar.markdown('## Step 3: Choose image')

        self.image_idx = st.sidebar.number_input(
            label='Please choose the index for the image here (choose -1 to show 6 random images)', 
            min_value=-1, max_value=self.num_images, value=0, step=1,
            format='%d'
        )

        if (self.image_idx == -1):
            self.image_idx = None
    
    def _show_images(self):

        if self.image_idx is None:
            
            # choose 6 random images
            show_idxs = random.sample(list(range(self.num_images)), 6)

            for idx in show_idxs:

                st.sidebar.image(f'assets/images/{idx}.jpg',use_column_width=True,caption=idx)
            
        else:
            st.sidebar.image(f'assets/images/{self.image_idx}.jpg',use_column_width=True,caption=self.image_idx)


    def _fix_model_name(self, model_name):

        if ('MFB' in model_name):
            self.model_name = 'mfb'
        
        elif ('MCAN' in model_name):
            self.model_name = 'mcan'
        
        else:
            raise NotImplementedError
