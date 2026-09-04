from pathlib import Path

from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from app.model import load_model
from app.prediction import predict_image


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "busi_cnn.pth"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = None
        self.device = None

        self.setWindowTitle("Breast Cancer Detector")
        self.setMinimumSize(900, 700)

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(18)

        title = QLabel("Breast Cancer Detector")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size: 28px; font-weight: bold; padding: 10px;"
        )
        layout.addWidget(title)

        subtitle = QLabel(
            "BUSI CNN • Benign / Malignant / Normal"
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 15px;")
        layout.addWidget(subtitle)

        self.image_label = QLabel("Select a breast ultrasound image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 420)
        self.image_label.setStyleSheet(
            "border: 2px dashed #888; border-radius: 12px; "
            "font-size: 17px; padding: 15px;"
        )
        layout.addWidget(self.image_label, stretch=1)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        self.open_button = QPushButton("Select Image")
        self.open_button.setMinimumHeight(45)
        self.open_button.clicked.connect(self._select_image)
        button_layout.addWidget(self.open_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setMinimumHeight(45)
        self.clear_button.clicked.connect(self._clear)
        button_layout.addWidget(self.clear_button)

        layout.addLayout(button_layout)

        self.result_label = QLabel("Prediction: —")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet(
            "font-size: 23px; font-weight: bold; padding: 12px;"
        )
        layout.addWidget(self.result_label)

        self.confidence_label = QLabel("Confidence: —")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet("font-size: 17px;")
        layout.addWidget(self.confidence_label)

        self.status_label = QLabel("Model: loading...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.status_label)

    def _load_model(self):
        try:
            self.model, self.device = load_model(MODEL_PATH)
            self.status_label.setText(
                f"Model loaded successfully • Device: {self.device}"
            )
        except Exception as error:
            self.status_label.setText("Model: not loaded")
            self.open_button.setEnabled(False)

            QMessageBox.warning(
                self,
                "Model not found",
                str(error),
            )

    def _select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Breast Ultrasound Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*.*)",
        )

        if not file_path:
            return

        try:
            self._show_image(file_path)

            prediction, confidence, probabilities = predict_image(
                self.model,
                self.device,
                file_path,
            )

            self.result_label.setText(
                f"Prediction: {prediction.upper()}"
            )
            self.confidence_label.setText(
                f"Confidence: {confidence * 100:.1f}%"
            )

            details = " | ".join(
                f"{name}: {probability * 100:.1f}%"
                for name, probability in probabilities.items()
            )
            self.status_label.setText(details)

        except Exception as error:
            QMessageBox.critical(
                self,
                "Prediction Error",
                str(error),
            )

    def _show_image(self, file_path):
        pixmap = QPixmap(file_path)

        if pixmap.isNull():
            raise ValueError("The selected file is not a valid image.")

        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.image_label.setPixmap(scaled_pixmap)

    def _clear(self):
        self.image_label.clear()
        self.image_label.setText("Select a breast ultrasound image")
        self.result_label.setText("Prediction: —")
        self.confidence_label.setText("Confidence: —")

        if self.model is not None:
            self.status_label.setText(
                f"Model loaded successfully • Device: {self.device}"
            )


def run_app():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
