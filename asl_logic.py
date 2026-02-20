import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import copy
import itertools
from PyQt5.QtCore import QThread, pyqtSignal

class ASLThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    text_detected_signal = pyqtSignal(str)

    def __init__(self, model_path, label_path):
        super().__init__()
        self.running = False
        self.cam_index = 0
        self.model_path = model_path
        self.label_path = label_path
        self.model = None
        self.classes = []

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.history_buffer = []

    def load_models(self):
        """Загрузка моделей внутри потока (важно для TensorFlow)"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            with open(self.label_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"ASL Model Error: {e}")

    def calc_landmark_list(self, image, landmarks):
        w, h = image.shape[1], image.shape[0]
        return [[min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)] for lm in landmarks.landmark]

    def pre_process_landmark(self, landmark_list):
        temp = copy.deepcopy(landmark_list)
        base_x, base_y = temp[0][0], temp[0][1]
        for i, pt in enumerate(temp):
            temp[i][0] -= base_x
            temp[i][1] -= base_y
        return list(itertools.chain.from_iterable(temp))

    def run(self):
        self.load_models()
        if not self.model: return
        self.running = True
        cap = cv2.VideoCapture(self.cam_index)
        while self.running:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)
            current_char = ""
            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    pre = self.pre_process_landmark(self.calc_landmark_list(frame, hl))
                    pred = self.model.predict(np.array([pre], dtype=np.float32), verbose=0)
                    idx = np.argmax(pred)
                    if pred[0][idx] > 0.7: current_char = self.classes[idx]
                    mp.solutions.drawing_utils.draw_landmarks(frame, hl, self.mp_hands.HAND_CONNECTIONS)

            if current_char:
                self.history_buffer.append(current_char)
                if len(self.history_buffer) > 15: self.history_buffer.pop(0)
                if self.history_buffer.count(current_char) == 15:
                    self.text_detected_signal.emit(current_char)
                    self.history_buffer = []
            self.change_pixmap_signal.emit(frame)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()