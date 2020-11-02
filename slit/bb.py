import streamlit as st
import numpy as np
import cv2

class BoundingBox():

    def __init__(self, image: np.ndarray, bboxes: np.ndarray):
        """[summary]

        Args:
            image (np.ndarray): shape(height, width, channels)
            bboxes (np.ndarray): shape(n_heads, num_objects, 5)
        """
        
        # [] slider
        st.markdown('### Top-attended bounding boxes')
        st.markdown(
            'To answer the question, the system pays more attention to some regions of the image\
            than others. The boxes here depict those regions.\n\n'
            'The green boxes denote the regions with > 10% attention, blue boxes\
            with attention score between 5% to 10% and the red boxes with less than 5%.'
        )

        self.cols = st.beta_columns(bboxes.shape[0])
        self.button_cols = st.beta_columns(3)

        self.image = np.copy(image)
        self.bboxes = bboxes

        with self.button_cols[0]:
            self.show_green = st.checkbox(
                label='Show green objects', value=True, key='show_green'
        )
        with self.button_cols[1]:
            self.show_blue = st.checkbox(
                label='Show blue objects', value=True, key='show_blue'
        )
        with self.button_cols[2]:
            self.show_red = st.checkbox(
                label='Show red objects', value=False, key='show_red'
        )

        self.confidence_th = st.slider(
            label='Only objects with >= than this threshold will be shown', 
            min_value=0.0, max_value=1.0, value=0.07, step=0.01
        )

        self.colors = {
            "green": [0, 255, 0],
            "blue": [0, 0, 170],
            "red": [100, 0, 0]
        }

        self.is_shown = {
            "green": self.show_green,
            "blue": self.show_blue,
            "red": self.show_red
        }

        self.plot_boxes()

    
    def plot_boxes(self):

        image_with_boxes = self.image.astype(np.float64)
        
        height, width, channels = image_with_boxes.shape

        for idx in range(self.bboxes.shape[0]):

            for (xmin, ymin, xmax, ymax, confidence) in self.bboxes[idx]:

                if (confidence >= self.confidence_th):

                    if (confidence <= 0.05):
                        cat = "red"
                    elif (0.05 < confidence <= 0.1):
                        cat = "blue"
                    else:
                        cat = "green"

                    if (not self.is_shown[cat]):
                        continue

                    image_with_boxes[int(ymin*height):int(ymax*height),int(xmin*width):int(xmax*width),:] += self.colors[cat]
                    image_with_boxes[int(ymin*height):int(ymax*height),int(xmin*width):int(xmax*width),:] /= 2
                    cv2.putText(
                        img=image_with_boxes, text=f'{confidence*100:.2f}%', org=(int(xmin*width),int(ymin*height)), color=(0,0,0),
                        fontFace=0, fontScale=0.8, thickness=2
                    )
                    
            with self.cols[idx]:
                st.image(
                    image_with_boxes.astype(np.uint8), use_column_width=True,
                    caption=f'Prediction #{idx}'
                )
            
        




