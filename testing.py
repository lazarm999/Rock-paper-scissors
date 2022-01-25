import cv2
import numpy as np

import os
from keras.models import load_model
from train import data_generator, dim
from RPS_game import get_prediction_text, font_size

def test_model_with_data(test_dir, model_name='model'):
    model_path = os.path.join('models', f'{model_name}.h5')
    model = load_model(model_path)
    gen = data_generator(test_dir, batch_size=1, validation=True)

    n = 0
    l = len(gen)
    correct = 0
    for X, y in gen:
        if n == l: break
        n+=1
        predictions = model.predict(X)[0]
        y = y[0]
        pIndeks = np.argmax(predictions)
        trueIndeks = np.argmax(y)
        if pIndeks == trueIndeks:
            correct+=1
        print('T' if pIndeks == trueIndeks else 'F', 'prediction:[', round(predictions[0] * 100, 2),
            round(predictions[1] * 100, 2), round(predictions[2] * 100, 2), '] true value: ', y)
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
        prediction = model.predict(np.divide([cv2.resize(frame, (dim, dim), interpolation=cv2.INTER_CUBIC)], 255))
        prediction_text = get_prediction_text(prediction)
        cv2.putText(frame, prediction_text, (int(cam_w / 2), 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (225, 0, 0))
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='This module is used for testing the Rock, paper, scissors model either with pictures from disk or with camera.')
    parser.add_argument('-d', '--dir', type=str, default='', help='The path to your validation data directory, \
        which has 3 subdirectories correspoding to the rock, paper and scissors class. This argument must be passed if the -c flag is not set.')
    parser.add_argument('-c', '--camera', action='store_true', help='Whether to use the camera for testing.')
    parser.add_argument('-m', '--model', type=str, default='model', help='Name of the model in h5 format in the models directory\
        to load for testing, defaults to \'model\'.')
    
    args = parser.parse_args()
    if args.camera:
        test_model_with_camera(model_name=args.model)
    elif args.dir:
        test_model_with_data(args.dir, model_name=args.model)
    else:
        print('Either the -c or -d parameter must be set.')