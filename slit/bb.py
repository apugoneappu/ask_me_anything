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
            than others. The white boxes on the image depict those regions.  \n ')
        st.markdown(
            'However, we may end up with an incorrect answer if some critical regions are not attended to. Thus, the model makes two sets of predictions to avoid missing such important regions.'
        )

        self.cols = st.beta_columns(bboxes.shape[0])

        self.image = np.copy(image)
        self.bboxes = bboxes

        self.topk = st.number_input(
            'Adjust the value below from 1 to 10 to see the top-10 objects in decreasing order of confidence.', 
            min_value=1, max_value=10, value=1, step=1, format=None, key=None
        )

        self.plot_boxes()

    
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

            image_with_boxes = self.image.astype(np.float64)
            height, width, channels = image_with_boxes.shape

            for obj_idx, (xmin, ymin, xmax, ymax, confidence) in enumerate(self.bboxes[idx]):

                if (obj_idx+1 != self.topk):
                    continue
                
                # Mask is true everywhere else except the object
                mask = np.ones_like(image_with_boxes)

                # Mask is false on the object
                mask[int(ymin):int(ymax),int(xmin):int(xmax),:] = 0

                mask = mask.astype(np.bool)

                # Darken the non-object regions
                image_with_boxes[mask] /= 3

                self.plot_box(image_with_boxes, xmin, ymin, xmax, ymax, height, width)

                # Write the accuracy on the top-left corner
                # cv2.putText(
                #     img=image_with_boxes, text=f'{confidence*100:.2f}%', org=(10,30), color=(255,255,255),
                #     fontFace=0, fontScale=1, thickness=2
                # )

                with self.cols[idx]:
                    st.write(f'Confidence: {100 * confidence:.4f}')
                    
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

        




