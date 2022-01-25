import os
import cv2
import numpy as np
import threading

from keras.models import load_model
from playsound import playsound

from utils import dim, get_prediction_text

game_started = False
font_size = 1
last_winner = ""
color = (255, 0, 0)

def play_game(model_name='model'):
    global game_started
    global last_winner
    global color
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('test', 1080, 720)
    img_counter = 1
    model_path = os.path.join('models', f'{model_name}.h5')
    model = load_model(model_path)
    cam_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) #640
    cam_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) #480
    while True:
        if img_counter == 5000:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            frame = cv2.flip(frame, 1)
            org_frame = frame
            if game_started:
                p1_hand = frame[0:cam_h, 0:int(cam_w / 2)]
                p2_hand = frame[0:479, int(cam_w / 2):cam_w]
                p1_hand = [cv2.resize(p1_hand, (dim, dim), interpolation=cv2.INTER_CUBIC)]
                p2_hand = [cv2.resize(p2_hand, (dim, dim), interpolation=cv2.INTER_CUBIC)]
                p1_prediction = model.predict(np.divide(p1_hand, 255))
                p2_prediction = model.predict(np.divide(p2_hand, 255))
                p1_prediction_text = get_prediction_text(p1_prediction)
                p2_prediction_text = get_prediction_text(p2_prediction)
                if (p1_prediction_text == "papir" and p2_prediction_text == "kamen") or \
                        (p1_prediction_text == "makaze" and p2_prediction_text == "papir") or \
                        p1_prediction_text == "kamen" and p2_prediction_text == "makaze":
                    last_winner = "p1"
                elif (p1_prediction_text == "papir" and p2_prediction_text == "papir") or \
                        (p1_prediction_text == "makaze" and p2_prediction_text == "makaze") or \
                        p1_prediction_text == "kamen" and p2_prediction_text == "kamen":
                    last_winner = "nereseno"
                else:
                    last_winner = "p2"
                game_started = False

            if last_winner == "p1":
                cv2.putText(org_frame, "Pobednik", (int(cam_w / 5), 50),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, color)
            elif last_winner == "p2":
                cv2.putText(org_frame, "Pobednik", (int(0.7 * cam_w), 50),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, color)
            elif last_winner == "nereseno":
                cv2.putText(org_frame, "Nereseno", (int(0.42   * cam_w), 50),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, color)
            cv2.line(img=org_frame, pt1=(int(cam_w/2), 75), pt2=(int(cam_w/2), cam_h),
                     color=(255, 0, 0), thickness=1, lineType=8, shift=0)
            cv2.imshow("test", org_frame)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                print("Escape hit, closing...")
                break
            if k % 256 == 32:
                print("Space hit")
                timer = threading.Timer(6.0, start_game)
                threading.Thread(target=lambda: playsound('rps_sound.wav')).start()
                timer.start()
            img_counter = 1
        else:
            img_counter = img_counter + 1
    cam.release()
    cv2.destroyAllWindows()

def start_game():
    global game_started
    global color
    game_started = True
    if color == (255, 0, 0):
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Start the game.')
    parser.add_argument('-m', '--model', type=str, default='model', help='Name of the model in h5 format in the models directory \
        to load for the game, defaults to \'model\'.')
    
    args = parser.parse_args()
    play_game(model_name=args.model)