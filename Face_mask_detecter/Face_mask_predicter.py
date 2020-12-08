import tensorflow as tf
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np

model = tf.keras.models.load_model('saved_model/my_model')

font_scale = 1
thickness = 2
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
# define a video capture object
vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    img = tf.keras.preprocessing.image.img_to_array(frame)
    detector = MTCNN()
    faces = detector.detect_faces(img)

    for f in faces:
        (x, y, w, h) = f['box']
        croped_img = frame[y:y + h, x:x + w]
        face = tf.keras.preprocessing.image.array_to_img(croped_img).resize((224, 224), 0)
        face = tf.keras.preprocessing.image.img_to_array(face)

        face = np.expand_dims(face, axis=0)

        classes = model.predict(face)

        if classes[0][0] == 0:
            cv2.putText(frame, "Masked", (x, y - 10), font, font_scale, green, thickness)
            cv2.rectangle(frame, (x, y), (x + w, y + h), green, 2)
        else:
            cv2.putText(frame, "No Mask", (x, y - 10), font, font_scale, red, thickness)
            cv2.rectangle(frame, (x, y), (x + w, y + h), red, 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()