import cv2
import numpy as np
import random
from PIL import Image

class Augmentation(object):
    def __init__(self, prob=0.4):
        self.prob = prob

    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    def translate_image(self, image, t_x, t_y):
        rows, cols = image.shape[:2]
        M = np.array([[1, 0, t_x], [0, 1, t_y]], dtype=np.float32)
        return cv2.warpAffine(image, M, (cols, rows))

    def __call__(self, img):
        image = np.array(img)

        # 1. Rotation (-15 to 15)
        if random.random() < self.prob:
            angle = random.uniform(-15, 15)
            image = self.rotate_image(image, angle)

        # 2. Scaling (max 10% zoom in/out)
        if random.random() < self.prob:
            scale = random.uniform(0.9, 1.1)
            h, w = image.shape[:2]
            resized = cv2.resize(image, None, fx=scale, fy=scale)
            canvas = np.zeros_like(image)
            
            # Crop or pad 
            new_h, new_w = resized.shape[:2]
            y_start = max(0, (new_h - h) // 2)
            x_start = max(0, (new_w - w) // 2)
            cropped = resized[y_start:y_start+h, x_start:x_start+w]
            
            cy, cx = cropped.shape[:2]
            py = max(0, (h - cy) // 2)
            px = max(0, (w - cx) // 2)
            canvas[py:py+cy, px:px+cx] = cropped
            image = canvas

        # 3. Translation (Shift max 3 pixels)
        if random.random() < self.prob:
            t_x = random.randint(-3, 3)
            t_y = random.randint(-3, 3)
            image = self.translate_image(image, t_x, t_y)

        # 4. Morphological (erosion/dilation)
        if random.random() < self.prob:
            k_size = random.choice([2, 3])
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            if random.random() < 0.5:
                image = cv2.erode(image, se, iterations=1)
            else:
                image = cv2.dilate(image, se, iterations=1)

        # 5. Cutout / Random Erasing
        if random.random() < self.prob:
            h, w = image.shape[:2]
            size = random.randint(4, 8)
            y = random.randint(0, max(1, h - size))
            x = random.randint(0, max(1, w - size))
            image[y:y+size, x:x+size] = 0

        return Image.fromarray(image)
