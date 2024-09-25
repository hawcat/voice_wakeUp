import os
import sys
import wave
from collections import deque
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import scipy.io.wavfile
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from tensorflow import keras


def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))  # 窗口中采样点的个数
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = scipy.signal.spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
    )

    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


class AudioInferenceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initAudio()
        self.loadModel()
        self.commands = [
            "_background_noise_",
            "go",
            "max",
            "no",
            "off",
            "on",
            "stop",
            "wow",
        ]
        self.background_thread = None

    def initUI(self):
        self.setWindowTitle("Realtime Audio Inference")
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        self.recordButton = QPushButton("Start Recording", self)
        self.recordButton.clicked.connect(self.toggleRecording)
        layout.addWidget(self.recordButton)

        self.resultLabel = QLabel("Inference Result: ", self)
        self.recordLabel = QLabel("Record status", self)
        layout.addWidget(self.resultLabel)
        layout.addWidget(self.recordLabel)

        # 创建Figure和Canvas用于显示图表
        self.figure = plt.figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def initAudio(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 1

        self.p = pyaudio.PyAudio()
        self.stream = None
        self.frames = []

        self.is_recording = False

    def loadModel(self):
        # 加载预训练模型
        self.model = keras.models.load_model("best_model_max.keras")

    def toggleRecording(self):
        if not self.is_recording:
            self.startRecording()
        else:
            self.stopRecording()

    def startRecording(self):
        self.is_recording = True
        self.recordButton.setText("Stop Recording")
        self.background_thread = Thread(target=self.detect_sound_and_record)
        self.background_thread.start()

    def stopRecording(self):
        self.is_recording = False
        self.recordButton.setText("Start Recording")
        self.recordLabel.setText("Recording stopped")

    def detect_sound_and_record(
        self,
        threshold=1000,
        chunk_size=1024,
        record_seconds=1,
        pre_buffer_seconds=0.2,
        sample_rate=16000,
    ):
        while self.is_recording:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
            )

            self.recordLabel.setText("Waiting for sound...")

            pre_buffer = deque(
                maxlen=int(sample_rate / chunk_size * pre_buffer_seconds)
            )

            while self.is_recording:
                data = stream.read(chunk_size, exception_on_overflow=False)
                pre_buffer.append(data)
                audio_data = np.frombuffer(data, dtype=np.int16)
                if np.abs(audio_data).mean() > threshold:
                    self.recordLabel.setText("Sound detected! Recording...")
                    break

            frames = list(pre_buffer)
            for _ in range(0, int(sample_rate / chunk_size * record_seconds)):
                if not self.is_recording:
                    break
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)

            self.recordLabel.setText("Recording finished. Check output.wav")

            stream.stop_stream()
            stream.close()
            p.terminate()

            # Save recording
            wf = wave.open("output.wav", "wb")
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
            wf.close()

            if os.path.isfile("output.wav"):
                self.inference("output.wav")

            if not self.is_recording:
                break

    def get_specgram(self, file_path):
        sample_rate, signal = scipy.io.wavfile.read(file_path)
        signal_padding = np.zeros((16000,))
        if len(signal) >= 16000:
            signal_padding = signal[:16000]
        else:
            signal_padding[: len(signal)] = signal
        _, _, specgram = log_specgram(signal_padding, sample_rate=sample_rate)
        return specgram

    def inference(self, file_path):
        specgram = self.get_specgram(file_path)
        result = self.model.predict(specgram.reshape(-1, 99, 161))
        y_pred = np.argmax(result)
        data = result.reshape(8)  # 假设有8个命令类别

        self.resultLabel.setText(f"Predicted Command: {self.commands[y_pred]}")

        # 更新图表
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.bar(self.commands, data)
        ax.set_ylabel("Probability")
        ax.set_title("Command Prediction")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = AudioInferenceApp()
    ex.show()
    sys.exit(app.exec_())
