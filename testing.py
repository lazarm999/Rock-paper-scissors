import cv2
import numpy as np

import os
from keras.models import load_model
from train import data_generator, width, height
from RPS_game import get_prediction_text, font_size

def test_model_with_data(test_dir, model_name='model'):
    model_path = os.path.join('models', f'{model_name}.h5')
    model = load_model(model_path)
    gen = data_generator(test_dir, validation=True)

    n = 0
    for X, y in gen:
        predictions = model.predict(X)
        correct = 0
        n += len(predictions)
        for i in range(len(predictions)):
            pIndeks = np.argmax(predictions[i])
            trueIndeks = np.argmax(y[i])
            if pIndeks == trueIndeks:
                correct+=1
            print('T' if pIndeks == trueIndeks else 'F', 'prediction:[', round(predictions[i][0] * 100, 2),
                round(predictions[i][1] * 100, 2), round(predictions[i][2] * 100, 2), '] true value: ', y[i])
    print('total:', n, ' correct:', correct, ' wrong:', n-correct)

def test_model_with_camera(model_name='model'):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('test', 1080, 720)
    model_path = os.path.join('models', f'{model_name}.h5')
    model = load_model(model_path)

    cam_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))  # 640
    cam_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 480

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        frame = cv2.flip(frame, 1)
        prediction = model.predict(np.divide([cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)], 255))
        prediction_text = get_prediction_text(prediction)
        cv2.putText(frame, prediction_text, (int(cam_w / 2), 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (225, 0, 0))
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
    cam.release()
    cv2.destroyAllWindows()

