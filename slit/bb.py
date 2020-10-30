import streamlit as st
import numpy as np
import cv2

class BoundingBox():

    def __init__(self, image: np.ndarray, bboxes: list):
        
        # [] slider
        
        self.image = np.copy(image)
        self.bboxes = bboxes

        self.show_all = st.checkbox(
            label='Show/Hide all objects', value=True, key='show_all'
        )
        self.confidence_th = st.slider(
            label='Only objects with >= than this threshold will be shown', 
            min_value=0.0, max_value=1.0, value=0.0, step=0.01
        )

        self.colors = {
            "max": [0, 255, 0],
            "mid": [0, 0, 170],
            "min": [100, 0, 0]
        }

        self.plot_boxes()

    
    def plot_boxes(self):

        image_with_boxes = self.image.astype(np.float64)
        
        height, width, channels = image_with_boxes.shape

        if self.show_all:

            for (xmin, xmax, ymin, ymax, confidence) in self.bboxes:

                if (confidence >= self.confidence_th):

                    if (confidence <= 0.05):
                        cat = "min"
                    elif (0.05 < confidence <= 0.1):
                        cat = "mid"
                    else:
                        cat = "max"

                    image_with_boxes[int(ymin*height):int(ymax*height),int(xmin*width):int(xmax*width),:] += self.colors[cat]
                    image_with_boxes[int(ymin*height):int(ymax*height),int(xmin*width):int(xmax*width),:] /= 2
                    cv2.putText(
                        img=image_with_boxes, text=f'{confidence*100}%', org=(int(xmin*width),int(ymin*height)), color=self.colors[cat],
                        fontFace=0, fontScale=1.3, thickness=2
                    )
                

        st.image(
            image_with_boxes.astype(np.uint8), use_column_width=True,
            caption='Image with bounding boxes (Green > 10%, 5% < blue < 10%, red < 5%)'
        )
            
        




