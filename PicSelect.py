import sys
import os
import cv2
import glob
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QMessageBox, QFileDialog, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


import sys
import os
import cv2
import glob
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QMessageBox, QFileDialog, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QMessageBox, QFileDialog, QCheckBox, QTextEdit
)

class ImageSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片筛选器")

        # 默认路径
        self.image_dir = ""
        self.save_dir_a = ""
        self.save_dir_d = ""
        self.image_paths = []
        self.index = 0

        # 顶部快捷键提示
        self.help_label = QLabel(
            "快捷键说明： W - 保存到目录A  |  F - 保存到目录D  |  D - 下一张  |  A - 上一张  |  R - 删除  |  Esc - 退出"
        )
        self.help_label.setStyleSheet("font-size: 14px; color: blue;")

        # 图片显示
        self.label = QLabel("请选择图片目录", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: 1px solid gray;")
        self.label.setMinimumSize(1280, 720)

        # 复选框：是否删除原图
        self.checkbox_delete = QCheckBox("保存后删除原图")

        # === 左边：路径选择布局 ===
        path_layout = QVBoxLayout()

        # 图片目录
        row_image = QHBoxLayout()
        btn_select_image = QPushButton("图像目录")
        btn_select_image.setFixedWidth(80)
        self.path_label_image = QLabel("未选择")
        btn_select_image.clicked.connect(self.select_image_dir)
        row_image.addWidget(btn_select_image)
        row_image.addWidget(self.path_label_image)
        path_layout.addLayout(row_image)

        # 保存目录A
        row_a = QHBoxLayout()
        btn_select_a = QPushButton("保存A")
        btn_select_a.setFixedWidth(80)
        self.path_label_a = QLabel("未选择")
        btn_select_a.clicked.connect(self.select_save_a)
        row_a.addWidget(btn_select_a)
        row_a.addWidget(self.path_label_a)
        path_layout.addLayout(row_a)

        # 保存目录D
        row_d = QHBoxLayout()
        btn_select_d = QPushButton("保存D")
        btn_select_d.setFixedWidth(80)
        self.path_label_d = QLabel("未选择")
        btn_select_d.clicked.connect(self.select_save_d)
        row_d.addWidget(btn_select_d)
        row_d.addWidget(self.path_label_d)
        path_layout.addLayout(row_d)

        # === 右边：日志区 ===
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("font-size: 13px; background: #DADCE3; color: black;")


        # 底部分成左右
        bottom_layout = QHBoxLayout()
        bottom_layout.addLayout(path_layout, 2)  # 左边
        bottom_layout.addWidget(self.log_box, 3) # 右边

        # === 总布局 ===
        layout = QVBoxLayout()
        layout.addWidget(self.help_label)   # 顶部提示
        layout.addWidget(self.label)        # 图片显示
        layout.addWidget(self.checkbox_delete)
        layout.addLayout(bottom_layout)     # 底部：路径 + 日志
        self.setLayout(layout)

    # ========== 日志函数 ==========
    def log(self, text):
        self.log_box.append(text)
        self.log_box.ensureCursorVisible()

    def select_image_dir(self):
        self.image_dir = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if self.image_dir:
            self.path_label_image.setText(f"图片目录: {self.image_dir}")
            self.load_images()

    def select_save_a(self):
        self.save_dir_a = QFileDialog.getExistingDirectory(self, "选择保存目录A")
        if self.save_dir_a:
            os.makedirs(self.save_dir_a, exist_ok=True)
            self.path_label_a.setText(f"保存目录A: {self.save_dir_a}")

    def select_save_d(self):
        self.save_dir_d = QFileDialog.getExistingDirectory(self, "选择保存目录D")
        if self.save_dir_d:
            os.makedirs(self.save_dir_d, exist_ok=True)
            self.path_label_d.setText(f"保存目录D: {self.save_dir_d}")

    def load_images(self):
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.*")))
        self.image_paths = [p for p in self.image_paths if p.lower().endswith((".JPG", ".jpg", ".jpeg", ".png"))]
        self.index = 0
        self.show_image()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_W:
            if self.save_dir_a:
                self.save_image(self.save_dir_a, "A")
                if self.checkbox_delete.isChecked():
                    self.delete_image()
            else:
                QMessageBox.warning(self, "警告", "请先设置保存目录 A！")
        elif key == Qt.Key_F:
            if self.save_dir_d:
                self.save_image(self.save_dir_d, "D")
                if self.checkbox_delete.isChecked():
                    self.delete_image()
            else:
                QMessageBox.warning(self, "警告", "请先设置保存目录 D！")
        elif key == Qt.Key_D:
            self.log(f"跳过：{self.current_filename()}")
            self.index += 1
            self.show_image()
        elif key == Qt.Key_A:
            self.log(f"上一张：{self.current_filename()}")
            self.index = max(0, self.index - 1)
            self.show_image()
        elif key == Qt.Key_R:
            self.delete_image()
        elif key == Qt.Key_Escape:
            self.close()

    def show_image(self):
        if self.index >= len(self.image_paths):
            QMessageBox.information(self, "完成", "所有图片已处理完！")
            self.close()
            return

        path = self.image_paths[self.index]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.log(f"无法读取: {path}")
            self.index += 1
            self.show_image()
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.label.width(), self.label.height(), Qt.KeepAspectRatio
        )
        self.label.setPixmap(pixmap)
        self.setWindowTitle(
            f"[{self.index + 1}/{len(self.image_paths)}] {os.path.basename(path)}"
        )

    def current_filename(self):
        return os.path.basename(self.image_paths[self.index]) if self.image_paths else "无图片"

    def save_image(self, save_dir, tag):
        src_path = self.image_paths[self.index]
        filename = self.current_filename()
        dst_path = os.path.join(save_dir, filename)
        img = cv2.imdecode(np.fromfile(src_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imencode(os.path.splitext(dst_path)[1], img)[1].tofile(dst_path)
        self.log(f"{tag}类已保存: {dst_path}")
        self.index += 1
        self.show_image()

    def delete_image(self):
        path = self.image_paths[self.index]
        self.log(f"删除：{path}")
        try:
            os.remove(path)
        except Exception as e:
            self.log(f"删除失败：{e}")
        self.index += 1
        self.show_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageSelector()
    window.show()
    sys.exit(app.exec_())
