import keras
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2
#git

model = keras.models.load_model('fracture_detection.keras')

#test = 'x-data/data/val/fractured/1.jpg'
test = 'x-data/data/val/not fractured/1.jpg'
def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == "__main__":


    # image path
    img_path = test    # dog
    #img_path = '/media/data/dogscats/test1/19.jpg'      # cat

    # load a single image
    new_image = load_image(img_path)

    # check prediction
    pred = model.predict(new_image)

    print(pred)

