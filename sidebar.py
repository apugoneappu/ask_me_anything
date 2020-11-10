import streamlit as st
import random
from utils.count_files import num_images

class SideBar():

    def __init__(self) -> None:

        self.num_images = num_images()

        self.title = "Ask me anything (AMA)"
        self.model_name = None
        self.question = None
        self.image_idx = None

        self._title()
        self._model()
        self._question()
        self._image()

        self._show_images()

        self._show_author()
    
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
            value= 'What is the colour of the hat?',  
            key= 'question'
        )

    
    def _image(self):

        st.sidebar.markdown('## Step 3: Choose image')

        self.image_idx = st.sidebar.number_input(
            label='Please choose the index for the image here (choose -1 to show 6 random images). The model has not been trained on these images.', 
            min_value=-1, max_value=self.num_images, value=1, step=1,
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

    def _show_author(self):

        st.sidebar.markdown(
            '## Future releases'
        )

        st.sidebar.info(
            "Version 2 of this is already underway with -  \n • custom image uploads  \n • many more models!  \n "
            "To stay tuned for future releases, follow me on Twitter.  \n"
            "Please consider starring this repo if you like it!"
        )

        cols = st.sidebar.beta_columns((3,4))

        with cols[0]:
            st.components.v1.iframe(src="https://ghbtns.com/github-btn.html?user=apugoneappu&repo=ask_me_anything&type=star&count=true&size=large",
            height=30)

        with cols[1]:
            st.components.v1.iframe(src="https://platform.twitter.com/widgets/follow_button.html?screen_name=apoorve_singhal&show_screen_name=true&show_count=false&size=l",
            height=30)
        
        st.sidebar.markdown(
            '## About me'
        )
        st.sidebar.info(
            "Hi, I'm Apoorve. I like to explain the working of my networks with simple visualisations.  \n "
            "Please visit [apoorvesinghal.com](https://www.apoorvesinghal.com) if you wish to know more about me."
        )
