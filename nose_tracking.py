import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class NoseControlThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = False
        self.cam_index = 0

        self.smoothing = 12
        self.sensitivity = 2.0
        self.ear_threshold = 0.20
        self.double_blink_time = 0.4
        self.right_click_hold = 2.0
        self.exit_time = 5.0

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False

    def get_ear(self, landmarks, indices):
        def dist(p1, p2): return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

        A = dist(landmarks[indices[1]], landmarks[indices[5]])
        B = dist(landmarks[indices[2]], landmarks[indices[4]])
        C = dist(landmarks[indices[0]], landmarks[indices[3]])
        return (A + B) / (2.0 * C) if C != 0 else 0

    def run(self):
        self._run_flag = True
        cap = cv2.VideoCapture(self.cam_index)
        sw, sh = pyautogui.size()

        pyautogui.moveTo(sw // 2, sh // 2)
        prev_x, prev_y = sw // 2, sh // 2

        blink_counter = 0
        last_blink_time = 0
        was_closed = False
        eyes_closed_start_time = 0
        right_click_done = False

        while self._run_flag:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            current_time = time.time()

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                nose = lm[1]

                margin = 0.3 / self.sensitivity
                screen_x = np.interp(nose.x, (0.5 - margin, 0.5 + margin), (0, sw))
                screen_y = np.interp(nose.y, (0.5 - margin, 0.5 + margin), (0, sh))

                curr_x = prev_x + (screen_x - prev_x) / self.smoothing
                curr_y = prev_y + (screen_y - prev_y) / self.smoothing

                pyautogui.moveTo(int(curr_x), int(curr_y))
                prev_x, prev_y = curr_x, curr_y

                # Анализ глаз
                left_ear = self.get_ear(lm, [33, 160, 158, 133, 153, 144])
                right_ear = self.get_ear(lm, [362, 385, 387, 263, 373, 380])
                avg_ear = (left_ear + right_ear) / 2
                is_closed = avg_ear < self.ear_threshold

                if is_closed:
                    if eyes_closed_start_time == 0: eyes_closed_start_time = current_time
                    duration = current_time - eyes_closed_start_time

                    if duration >= self.right_click_hold and not right_click_done:
                        pyautogui.rightClick()
                        self.log_signal.emit("Action: Right Click (2s eyes closed)")
                        right_click_done = True

                    if duration >= self.exit_time:
                        self.log_signal.emit("System: Emergency Stop (5s eyes closed)")
                        break
                else:
                    eyes_closed_start_time = 0
                    right_click_done = False

                if is_closed and not was_closed:
                    if current_time - last_blink_time < self.double_blink_time:
                        blink_counter += 1
                    else:
                        blink_counter = 1
                    last_blink_time = current_time

                if blink_counter == 2 and not is_closed:
                    pyautogui.click()
                    self.log_signal.emit("Action: Left Click")
                    blink_counter = 0

                was_closed = is_closed

            self.change_pixmap_signal.emit(frame)

        cap.release()
        self._run_flag = False

    def stop(self):
        self._run_flag = False
        self.wait()