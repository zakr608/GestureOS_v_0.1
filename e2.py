import sys
import cv2
import json
import pyttsx3
import keyboard
import numpy as np
import sounddevice as sd
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# Функция для путей
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

from nose_tracking import NoseControlThread
from asl_logic import ASLThread
from vosk import Model, KaldiRecognizer

# ПУТИ
MODEL_PATH_RU = resource_path("vosk-model-small-ru-0.22")
MODEL_PATH_EN = resource_path("vosk-model-en-us-0.22-lgraph")
ASL_MODEL_FILE = resource_path("model.keras")
ASL_LABEL_FILE = resource_path("labels.txt")

KEY_LAYOUT = [
    [('ё', '`'), ('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'), ('6', '6'), ('7', '7'), ('8', '8'), ('9', '9'), ('0', '0'), ('-', '-'), ('=', '='), ('⌫', '⌫', 'func', 'backspace')],
    [('Tab', 'Tab', 'func', 'tab'), ('й', 'q'), ('ц', 'w'), ('у', 'e'), ('к', 'r'), ('е', 't'), ('н', 'y'), ('г', 'u'), ('ш', 'i'), ('щ', 'o'), ('з', 'p'), ('х', '['), ('ъ', ']'), ('\\', '\\')],
    [('Caps', 'Caps', 'func', 'caps lock'), ('ф', 'a'), ('ы', 's'), ('в', 'd'), ('а', 'f'), ('п', 'g'), ('р', 'h'), ('о', 'j'), ('л', 'k'), ('д', 'l'), ('ж', ';'), ('э', '\''), ('Enter', 'Enter', 'func', 'enter')],
    [('Shift', 'Shift', 'func', 'shift'), ('/', '/'), ('я', 'z'), ('ч', 'x'), ('с', 'c'), ('м', 'v'), ('и', 'b'), ('т', 'n'), ('ь', 'm'), ('б', ','), ('ю', '.'), ('.', '.'), ('Shift', 'Shift', 'func', 'shift')],
    [('Space', 'Space', 'func', 'space'), ('←', '←', 'func', 'left'), ('↑', '↑', 'func', 'up'), ('↓', '↓', 'func', 'down'), ('→', '→', 'func', 'right'), ('Lang', 'Lang', 'func', 'switch')]
]

class VirtualKeyboard(QWidget):
    def __init__(self):
        super().__init__()
        self.is_russian = True
        self.shift_active = False
        self.caps_active = False
        self.btn_map = []
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.resize(900, 320)
        self.old_pos = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        self.frame = QFrame()
        self.frame.setStyleSheet("background: #0f172a; border: 2px solid #38bdf8; border-radius: 15px;")
        layout.addWidget(self.frame)
        self.close_btn = QPushButton("×", self.frame); self.close_btn.setFixedSize(30, 30); self.close_btn.setFocusPolicy(Qt.NoFocus); self.close_btn.move(860, 10)
        self.close_btn.setStyleSheet("QPushButton { background: #ef4444; color: white; border-radius: 15px; font-weight: bold; font-size: 18px; border: none; } QPushButton:hover { background: #f87171; }")
        self.close_btn.clicked.connect(self.hide)
        grid = QGridLayout(); grid.setSpacing(6); grid.setContentsMargins(15, 50, 15, 15); self.frame.setLayout(grid)
        for r_idx, row in enumerate(KEY_LAYOUT):
            c_idx = 0
            for item in row:
                ru, en = item[0], item[1]
                type_ = item[2] if len(item) > 2 else 'char'; code = item[3] if len(item) > 3 else None
                btn = QPushButton(ru); btn.setFixedHeight(45); btn.setFocusPolicy(Qt.NoFocus)
                btn.setStyleSheet("QPushButton { background: #1e293b; color: white; border-radius: 8px; font-weight: bold; font-size: 13px; border: 1px solid #334155; } QPushButton:hover { background: #334155; }")
                btn.clicked.connect(lambda ch, t=type_, c=code, b=btn: self.on_click(t, c, b))
                span = 4 if ru == 'Space' else (2 if ru in ['Shift', 'Enter', 'Caps', 'Tab', 'Lang'] else 1)
                grid.addWidget(btn, r_idx, c_idx, 1, span); c_idx += span
                self.btn_map.append({'obj': btn, 'ru': ru, 'en': en, 'type': type_, 'code': code})
        self.update_keys_visuals()

    def update_keys_visuals(self):
        is_upper = self.shift_active or self.caps_active
        for b in self.btn_map:
            if b['type'] == 'char':
                text = b['ru'] if self.is_russian else b['en']
                b['obj'].setText(text.upper() if is_upper else text.lower())

    def on_click(self, type_, code, btn):
        try:
            if type_ == 'char':
                keyboard.write(btn.text())
                if self.shift_active: self.shift_active = False; self.update_keys_visuals()
            elif code == 'switch': self.is_russian = not self.is_russian; self.update_keys_visuals()
            elif code == 'shift': self.shift_active = not self.shift_active; self.update_keys_visuals()
            elif code == 'caps lock': self.caps_active = not self.caps_active; self.update_keys_visuals()
            elif code: keyboard.send(code)
        except: pass

    def mousePressEvent(self, e): self.old_pos = e.globalPos()
    def mouseMoveEvent(self, e):
        if self.old_pos:
            delta = e.globalPos() - self.old_pos; self.move(self.pos() + delta); self.old_pos = e.globalPos()

class VoiceThread(QThread):
    text_recognized = pyqtSignal(str); error_signal = pyqtSignal(str)
    def __init__(self, device_idx, model_path):
        super().__init__(); self.device_idx = device_idx; self.model_path = model_path; self.running = True
    def run(self):
        try:
            if not os.path.exists(self.model_path):
                self.error_signal.emit(f"Error: {self.model_path} not found"); return
            model = Model(self.model_path); rec = KaldiRecognizer(model, 16000)
            with sd.RawInputStream(samplerate=16000, blocksize=8000, device=self.device_idx, dtype='int16', channels=1) as stream:
                while self.running:
                    data, _ = stream.read(4000)
                    if rec.AcceptWaveform(bytes(data)):
                        res = json.loads(rec.Result())
                        if res['text']: self.text_recognized.emit(res['text'])
        except Exception as e: self.error_signal.emit(str(e))

class GestureOS(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GestureOS Pro")
        self.resize(1100, 750)
        self.tracker = NoseControlThread()
        self.asl_tracker = ASLThread(ASL_MODEL_FILE, ASL_LABEL_FILE)
        self.kb = VirtualKeyboard()
        self.voice_thread = None
        self.tracker.change_pixmap_signal.connect(self.update_image)
        self.asl_tracker.change_pixmap_signal.connect(self.update_image)
        self.asl_tracker.text_detected_signal.connect(self.handle_asl_text)
        self.init_ui(); self.apply_theme()

    def init_ui(self):
        central = QWidget(); self.setCentralWidget(central); layout = QHBoxLayout(central); layout.setContentsMargins(0, 0, 0, 0)
        sidebar = QFrame(); sidebar.setFixedWidth(220); sidebar.setStyleSheet("background:#0f172a;")
        side_lay = QVBoxLayout(sidebar); logo = QLabel("🤖 GestureOS"); logo.setStyleSheet("color:#38bdf8; font-size:22px; font-weight:bold; padding:20px;")
        self.btn_home = self.create_nav_btn("🏠 Home", True); self.btn_text = self.create_nav_btn("✍ Text & Talk", False)
        self.btn_settings = self.create_nav_btn("⚙ Settings", False); self.btn_about = self.create_nav_btn("ℹ About", False)
        side_lay.addWidget(logo); side_lay.addWidget(self.btn_home); side_lay.addWidget(self.btn_text); side_lay.addWidget(self.btn_settings); side_lay.addWidget(self.btn_about); side_lay.addStretch()
        self.pages = QStackedWidget(); self.pages.addWidget(self.create_home_page()); self.pages.addWidget(self.create_text_page()); self.pages.addWidget(self.create_settings_page()); self.pages.addWidget(self.create_about_page())
        layout.addWidget(sidebar); layout.addWidget(self.pages)
        self.btn_home.clicked.connect(lambda: self.switch_page(0)); self.btn_text.clicked.connect(lambda: self.switch_page(1)); self.btn_settings.clicked.connect(lambda: self.switch_page(2)); self.btn_about.clicked.connect(lambda: self.switch_page(3))

    def create_home_page(self):
        page = QWidget(); lay = QVBoxLayout(page)
        self.nose_card = self.create_mode_card("👃 Nose Mouse Control", "двойное моргание - ЛКМ, 2с глаза - ПКМ, 5с - выход.", "START")
        self.nose_card.findChild(QPushButton).clicked.connect(self.toggle_tracker)
        self.video_label = QLabel("Camera Stream Paused"); self.video_label.setAlignment(Qt.AlignCenter); self.video_label.setStyleSheet("background:black; border-radius:15px;"); self.video_label.setMinimumHeight(450)
        lay.addWidget(self.nose_card); lay.addWidget(self.video_label); return page

    def create_text_page(self):
        page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(30, 30, 30, 30)
        edit_tools = QHBoxLayout(); btn_q_copy = QPushButton("📋 Quick Copy"); btn_q_paste = QPushButton("📥 Quick Paste")
        for b in [btn_q_copy, btn_q_paste]: b.setStyleSheet("background: #334155; color: #38bdf8; font-weight: bold; border-radius: 5px; padding: 8px;"); edit_tools.addWidget(b)
        self.text_area = QTextEdit(); self.text_area.setStyleSheet("background:#1e293b; color:white; border-radius:12px; font-size:20px; padding:15px;")
        btn_q_copy.clicked.connect(lambda: QApplication.clipboard().setText(self.text_area.toPlainText()))
        btn_q_paste.clicked.connect(lambda: self.text_area.insertPlainText(QApplication.clipboard().text()))
        edit_tools.addStretch()
        stt_lay = QHBoxLayout(); self.btn_stt_ru = self.create_action_btn("🎙 Russian STT", "#1e293b"); self.btn_stt_en = self.create_action_btn("🎙 English STT", "#1e293b")
        self.btn_stt_ru.clicked.connect(lambda: self.toggle_stt(MODEL_PATH_RU, self.btn_stt_ru)); self.btn_stt_en.clicked.connect(lambda: self.toggle_stt(MODEL_PATH_EN, self.btn_stt_en))
        stt_lay.addWidget(self.btn_stt_ru); stt_lay.addWidget(self.btn_stt_en)
        btns = QGridLayout(); btn_speak = self.create_action_btn("🔊 Speak (TTS)", "#10b981"); btn_speak.clicked.connect(self.handle_tts); btn_kb = self.create_action_btn("⌨ Keyboard", "#38bdf8"); btn_kb.clicked.connect(lambda: self.kb.show())
        self.btn_asl = self.create_action_btn("🤟 ASL Mode", "#1e293b"); self.btn_asl.clicked.connect(self.toggle_asl)
        btns.addWidget(btn_speak, 0, 0); btns.addWidget(btn_kb, 0, 1); btns.addWidget(self.btn_asl, 1, 0, 1, 2)
        lay.addWidget(QLabel("<h1>Text & Communication</h1>")); lay.addLayout(edit_tools); lay.addWidget(self.text_area); lay.addLayout(stt_lay); lay.addLayout(btns); return page

    def create_settings_page(self):
        page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(40, 40, 40, 40); lay.addWidget(QLabel("<h1>Settings</h1>")); lay.addWidget(QLabel("Sensitivity:"))
        self.sens_label = QLabel(f"Current: {self.tracker.sensitivity:.1f}x"); self.sens_slider = QSlider(Qt.Horizontal); self.sens_slider.setRange(10, 50); self.sens_slider.setValue(int(self.tracker.sensitivity * 10))
        self.sens_slider.valueChanged.connect(self.update_sens); lay.addWidget(self.sens_label); lay.addWidget(self.sens_slider); lay.addSpacing(20)
        self.cam_box = QComboBox(); self.detect_cams(); lay.addWidget(QLabel("Camera:")); lay.addWidget(self.cam_box)
        self.mic_box = QComboBox(); self.detect_mics(); lay.addWidget(QLabel("Microphone:")); lay.addWidget(self.mic_box); lay.addStretch(); return page

    def create_about_page(self):
        page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(40, 40, 40, 40)
        about = QLabel("<h1>GestureOS Pro</h1><p>Система помощи людям с ОВЗ.</p><p>Автор: <b>Кравчук Гордей</b></p>"); about.setWordWrap(True); lay.addWidget(about); lay.addStretch(); return page

    def handle_asl_text(self, char):
        cursor = self.text_area.textCursor()
        if char == "del": cursor.deletePreviousChar()
        elif char == "space": self.text_area.insertPlainText(" ")
        elif char != "nothing": self.text_area.insertPlainText(char)

    def handle_tts(self):
        txt = self.text_area.toPlainText()
        if txt: eng = pyttsx3.init(); eng.setProperty('rate', 170); eng.say(txt); eng.runAndWait(); del eng

    def toggle_stt(self, path, btn):
        if self.voice_thread and self.voice_thread.isRunning():
            self.voice_thread.running = False; self.voice_thread.wait(); self.voice_thread = None
            self.btn_stt_ru.setStyleSheet(self.action_btn_style("#1e293b")); self.btn_stt_en.setStyleSheet(self.action_btn_style("#1e293b")); return
        self.voice_thread = VoiceThread(self.mic_box.currentData(), path)
        self.voice_thread.text_recognized.connect(lambda t: self.text_area.insertPlainText(t + " "))
        self.voice_thread.error_signal.connect(lambda e: QMessageBox.warning(self, "STT Error", e))
        self.voice_thread.start(); btn.setStyleSheet(self.action_btn_style("#ef4444"))

    def toggle_tracker(self):
        btn = self.nose_card.findChild(QPushButton)
        if not self.tracker.isRunning(): self.asl_tracker.stop(); self.tracker.cam_index = self.cam_box.currentData(); self.tracker.start(); btn.setText("STOP")
        else: self.tracker.stop(); btn.setText("START")

    def toggle_asl(self):
        if not self.asl_tracker.isRunning(): self.tracker.stop(); self.asl_tracker.cam_index = self.cam_box.currentData(); self.asl_tracker.start(); self.btn_asl.setText("🤟 STOP ASL")
        else: self.asl_tracker.stop(); self.btn_asl.setText("🤟 ASL Mode")

    def update_image(self, f):
        h, w, ch = f.shape; qi = QImage(f.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(qi).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def detect_cams(self):
        for i in range(2):
            c = cv2.VideoCapture(i);
            if c.isOpened(): self.cam_box.addItem(f"Camera {i}", i); c.release()

    def detect_mics(self):
        try:
            for i, d in enumerate(sd.query_devices()):
                if d['max_input_channels'] > 0: self.mic_box.addItem(d['name'], i)
        except: pass

    def update_sens(self, v): self.tracker.sensitivity = v / 10.0; self.sens_label.setText(f"Current: {self.tracker.sensitivity:.1f}x")
    def create_nav_btn(self, t, a):
        btn = QPushButton(t); btn.setCheckable(True); btn.setAutoExclusive(True); btn.setChecked(a); btn.setMinimumHeight(50); btn.setStyleSheet(self.nav_style()); return btn
    def create_mode_card(self, t, d, b):
        card = QFrame(); card.setStyleSheet("background:#1e293b; border-radius:15px; padding:20px;"); l = QVBoxLayout(card); l.addWidget(QLabel(f"<h3>{t}</h3>")); l.addWidget(QLabel(d)); btn = QPushButton(b); btn.setStyleSheet("background:#38bdf8; color:#0f172a; font-weight:bold; padding:12px; border-radius:10px;"); l.addWidget(btn); return card
    def create_action_btn(self, t, c): btn = QPushButton(t); btn.setMinimumHeight(55); btn.setStyleSheet(self.action_btn_style(c)); return btn
    def action_btn_style(self, c): return f"QPushButton {{ background:{c}; color:white; border-radius:10px; font-weight:bold; border: 1px solid #334155; }} QPushButton:hover {{ background:#334155; }}"
    def nav_style(self): return "QPushButton { background:transparent; color:#94a3b8; border:none; text-align:left; padding:15px; } QPushButton:hover { background:#1e293b; color:white; } QPushButton:checked { background:#38bdf8; color:#0f172a; font-weight:bold; }"
    def apply_theme(self): self.setStyleSheet("QMainWindow { background:#020617; } QLabel { color:#cbd5e1; } QComboBox { background:#1e293b; color:white; padding:5px; } QSlider::handle:horizontal { background: #38bdf8; border-radius: 5px; width: 15px; }")
    def switch_page(self, i): self.pages.setCurrentIndex(i)

if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyle("Fusion"); win = GestureOS(); win.show(); sys.exit(app.exec_())