import streamlit as st
import numpy as np
import cv2
import torch

class BoundingBox():

    def __init__(self, image: np.ndarray, bboxes: np.ndarray):
        """[summary]

        Args:
            image (np.ndarray): shape(height, width, channels)  
            bboxes (np.ndarray): shape(num_glimpses, num_boxes=20, 5)
                            where the 5 numbers (xmin, ymin, xmax, ymax, confidence)
        """
        
        st.markdown('### Top-attended bounding boxes')
        st.markdown(
            'To answer the question, the system pays more attention to some regions of the image\
            than others. The boxes on the image are colour-coded with the confidence of the system.')

        self.color_cols = st.beta_columns(4)

        with self.color_cols[0]:
            self.first = (st.color_picker('conf > 50%', '#48b5a3'))
        
        with self.color_cols[1]:
            self.second = (st.color_picker('25% < conf < 50%', '#6fb7d6'))

        with self.color_cols[2]:
            self.third = (st.color_picker('10% < conf < 25%', '#fca985'))

        with self.color_cols[3]:
            self.fourth = (st.color_picker('conf < 10%', '#f0e8cd'))
        
        st.markdown(
            'However, we may end up with an incorrect answer if some critical regions are not attended to. Thus, the model makes two sets of predictions to avoid missing such important regions.'
        )

        self.cols = st.beta_columns(bboxes.shape[0])

        self.image = np.copy(image)
        self.bboxes = bboxes

        self.topk = st.slider(
            'Drag the slider to change the number of objects (displayed in decreasing order of confidence)', 
            min_value=1, max_value=10, value=3, step=1, format=None, key=None
        )

        self.plot_boxes()

    def hex_to_rgb(self, hex):

        h = hex.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def get_top_bboxes(iatt_maps, bboxes, k=20):
        """Returns the padded bboxes sorted according to iatt_maps

        Args:
            iatt_maps ([type]): shape (num_glimpses, padded_num_objects)
            bboxes ([type]): shape (num_objects, 4)

        Returns:
            np.ndarray: shape(num_glimpses, k, 5)
        """

        padded = torch.zeros(100, 5)
        padded[:bboxes.shape[0],:4] = bboxes
        
        bbox_final = []
        for i in range(iatt_maps.shape[0]):

            padded[:,-1] = iatt_maps[i]
            _, indices = torch.topk(padded[:,-1], k)

            tmp = [ padded[idx].tolist() for idx in indices.detach().numpy()[:k]]
            bbox_final.append(tmp)
        
        return np.array(bbox_final)

    def plot_boxes(self):

        for idx in range(self.bboxes.shape[0]):

            image_with_boxes = np.copy(self.image.astype(np.float64))
            height, width, channels = image_with_boxes.shape
            total_mask = np.ones_like(image_with_boxes)

            for obj_idx, (xmin, ymin, xmax, ymax, confidence) in enumerate(self.bboxes[idx][:self.topk]):

                # Mask is true everywhere else except the object

                # Mask is false on the object
                total_mask[int(ymin):int(ymax),int(xmin):int(xmax),:] = 0

            total_mask = total_mask.astype(np.bool)
            image_with_boxes[total_mask] /= 2

            for obj_idx, (xmin, ymin, xmax, ymax, confidence) in enumerate(reversed(self.bboxes[idx][:self.topk])):

                if (confidence < 0.1):
                    col = self.hex_to_rgb(self.fourth)
                elif (confidence < 0.25):
                    col = self.hex_to_rgb(self.third)
                elif (confidence < 0.5):
                    col = self.hex_to_rgb(self.second)
                else:
                    col = self.hex_to_rgb(self.first)

                # Restore greyed out original
                image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] = self.image[int(ymin):int(ymax),int(xmin):int(xmax),:]/3
                image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax)] += 2*np.array(col)/3
                self.plot_box(image_with_boxes, xmin, ymin, xmax, ymax, height, width)

                # cv2.putText(
                #     img=image_with_boxes, text=f'{confidence*100:.2f}%', org=(int(xmin),int(ymin)+20), color=(255,255,255),
                #     fontFace=0, fontScale=0.8, thickness=2
                # )

            with self.cols[idx]:
                st.image(
                    image_with_boxes.astype(np.uint8), use_column_width=True,
                    caption=f'Prediction #{idx}'
                )
            
    def plot_box(self, img, xmin, ymin, xmax, ymax, height, width, line_width=3):
        
        def boundary(x, limit):

            if (x < 0):
                return 0
            
            if (x >= limit):
                return limit-1
            
            return x

        img[
            boundary(int(ymin)-line_width, height):boundary(int(ymin), height),
            boundary(int(xmin)-line_width, width):boundary(int(xmax)+line_width, width),
            :
        ] = 255

        img[
            boundary(int(ymax), height):boundary(int(ymax)+line_width, height),
            boundary(int(xmin)-line_width, width):boundary(int(xmax)+line_width, width),
            :
        ] = 255

        img[
            boundary(int(ymin)-line_width, height):boundary(int(ymax)+line_width, height),
            boundary(int(xmin)-line_width, width):boundary(int(xmin), width),
            :
        ] = 255

        img[
            boundary(int(ymin)-line_width, height):boundary(int(ymax)+line_width, height),
            boundary(int(xmax), width):boundary(int(xmax)+line_width, width),
            :
        ] = 255

        




