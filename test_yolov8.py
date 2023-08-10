# Preset: yolo_v8_m_pascalvoc
import numpy as np
import keras_core as keras
import keras_cv
from keras_cv import visualization

import cv2
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def draw_detections(image, predictions, threshold=0.5):
    """Draws detections on the image"""
    names = keras_cv.datasets.pascal_voc.get_class_names() # Get class names
   
    for box, conf, label in zip (predictions['boxes'][0], predictions['confidence'][0], predictions['classes'][0]):
        if conf >= threshold:
            print(box, conf, names[label])
            x, y, w, h = np.array(box).astype(int)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 1, 0), 2)
    return image

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def show_image(image):
    """Shows an image"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def save_image(image, filename):
    """Saves an image"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.0)
    plt.close()

# Class names from from http://host.robots.ox.ac.uk/pascal/VOC/ 
class_ids = ["Aeroplane",
             "Bicycle",
             "Bird",
             "Boat",
             "Bottle",
             "Bus",
             "Car",
             "Cat",
             "Chair",
             "Cow",
             "Dining Table",
             "Dog",
             "Horse",
             "Motorbike",
             "Person",
             "Potted Plant",
             "Sheep",
             "Sofa",
             "Train",
             "Tvmonitor",
             "Total",]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Test 1    -   Test with a single image
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

threshold = 0.6
filepath  = keras.utils.get_file(origin="https://i.imgur.com/gCNcJJI.jpg")
image     = np.array(keras.utils.load_img(filepath))
inference_resizing = keras_cv.layers.Resizing(640, 640, 
                                              pad_to_aspect_ratio=True, 
                                              bounding_box_format="xywh")
image_resized = inference_resizing([image])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
'''
model = keras_cv.models.YOLOV8Detector.from_preset("yolo_v8_m_pascalvoc",
                                                   bounding_box_format="xywh",)
'''

model = keras_cv.models.RetinaNet.from_preset("retinanet_resnet50_pascalvoc", 
                                                         bounding_box_format="xywh")

prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(bounding_box_format = "xywh",
                                                                 from_logits         = True,
                                                                 iou_threshold       = 0.5,
                                                                 confidence_threshold= threshold,)
model.prediction_decoder = prediction_decoder

predictions = model.predict(image_resized)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

visualization.plot_bounding_box_gallery(image_resized,
                                        value_range   = (0, 255),
                                        rows          = 1,
                                        cols          = 1,
                                        y_pred        = predictions,
                                        scale         = 5,
                                        font_scale    = 0.4,
                                        bounding_box_format="xywh",
                                        class_mapping=class_mapping,
                                    )

'''
import keras_core.ops as ops
image_annotations = np.array(ops.copy(image_resized[0,:,:,:])/255.0)
image_annotations = draw_detections(image_annotations, predictions)
save_image(image_annotations, "test_yolov8_1.jpg")
show_image(image_annotations)
'''

