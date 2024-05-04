import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QFileDialog
from PySide6.QtGui import QPixmap, QColor, QBrush
from PySide6.QtCore import Qt
from MainWindow import Ui_MainWindow
from ImageSegmentation import ImageSegmentation  # Подставьте ваш модуль здесь


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.action_download_img.triggered.connect(self.load_image)
        self.ui.btn_k_means.clicked.connect(self.segment_with_k_means)
        self.ui.btn_DBSCAN.clicked.connect(self.segment_with_dbscan)
        self.ui.btn_growing_seed.clicked.connect(self.segment_with_growing_seed)
        self.ui.btn_region_growin.clicked.connect(self.segment_with_region_growing)
        self.ui.btn_watershed.clicked.connect(self.segment_with_watershed)
        self.ui.btn_drop_color.clicked.connect(self.drop_color_clicked)

        self.ui.graph_view_img.mousePressEvent = self.get_pixel_color
        # Добавляем строку для установки красного цвета по умолчанию
        self.drop_color_clicked()
        self.image_segmentation = None
        self.selected_color = None

    def load_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.image_segmentation = ImageSegmentation(file_path)
            pixmap = QPixmap(file_path)
            scene = QGraphicsScene()
            item = QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.ui.graph_view_img.setScene(scene)
            self.ui.graph_view_img.fitInView(item, Qt.AspectRatioMode.KeepAspectRatio)

    def segment_with_k_means(self):
        if self.image_segmentation:
            num_clusters = int(self.ui.dspinbox_k_means.value())
            self.image_segmentation.compare_segmentations(
                self.image_segmentation.segment_image_kmeans,
                num_clusters=num_clusters, target_color=self.selected_color,
                method_name="Метод k-means", title1="Оригинальное изображение",
                title2="К-средних RGB", title3="К-средних CIE Lab"
            )

    def segment_with_dbscan(self):
        if self.image_segmentation:
            min_samples = int(self.ui.dspinbox_DBSCAN.value())
            self.image_segmentation.compare_segmentations(
                self.image_segmentation.segment_image_dbscan,
                min_samples=min_samples, target_color=self.selected_color,
                method_name="Метод DBSCAN", title1="Оригинальное изображение",
                title2="DBSCAN RGB", title3="DBSCAN CIE Lab"
            )

    def segment_with_growing_seed(self):
        if self.image_segmentation:
            seed_point = (int(self.ui.dspinbox_growing_seed.value()), int(self.ui.dspinbox_growing_seed.value()))
            threshold = int(self.ui.dspinbox_growing_seed.value())
            self.image_segmentation.compare_segmentations(
                self.image_segmentation.growing_seed_segmentation,
                seed_point=seed_point, threshold=threshold,
                method_name="Метод Growing seed", title1="Оригинальное изображение",
                title2="Growing seed RGB", title3="Growing seed CIE Lab"
            )

    def segment_with_region_growing(self):
        if self.image_segmentation:
            threshold = int(self.ui.dspinbox_region_growing.value())
            self.image_segmentation.compare_segmentations(
                self.image_segmentation.region_growing, threshold=threshold,
                method_name="Метод Region growing", title1="Оригинальное изображение",
                title2="Region growing RGB", title3="Region growing CIE Lab"
            )

    def segment_with_watershed(self):
        if self.image_segmentation:
            self.image_segmentation.compare_segmentations(
                self.image_segmentation.watershed_segmentation,
                method_name="Метод Water Shed", title1="Оригинальное изображение",
                title2="Water Shed RGB", title3="Water Shed CIE Lab"
            )

    def get_pixel_color(self, event):
        if self.image_segmentation:
            if self.ui.graph_view_img.scene() is not None:
                mapped_pos = self.ui.graph_view_img.mapToScene(event.pos())
                item = self.ui.graph_view_img.scene().itemAt(mapped_pos, self.ui.graph_view_img.transform())
                if isinstance(item, QGraphicsPixmapItem):
                    pixmap = item.pixmap()
                    if not pixmap.isNull():
                        pixel_color = pixmap.toImage().pixelColor(mapped_pos.toPoint())
                        self.selected_color = (pixel_color.red(), pixel_color.green(), pixel_color.blue())
                        self.update_color_view(pixel_color)

    def update_color_view(self, color):
        scene = QGraphicsScene()
        scene.addRect(0, 0, 50, 50, Qt.NoPen, QBrush(color))
        self.selected_color = color
        self.ui.graphicsView.setScene(scene)
        self.ui.graphicsView.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def drop_color_clicked(self):
        self.selected_color = None
        scene = QGraphicsScene()
        scene.addRect(0, 0, 50, 50, Qt.NoPen, QBrush(Qt.NoBrush))
        self.ui.graphicsView.setScene(scene)
        self.ui.graphicsView.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec())

# Указываем путь к изображению
# image_path = "images/321.jpg"
# image_path = "images/123.webp"
# image_path = "images/flower.jpg"
# image_segmentation = ImageSegmentation(image_path)
# color = (image_segmentation.RGB_image[200, 200], 200, 200)
# num_clusters = 12
# image_segmentation.compare_segmentations(image_segmentation.segment_image_kmeans,
#                                          num_clusters=num_clusters, target_color=color,
#                                          method_name="Метод k-means", title1="Оригинальное изображение",
#                                          title2="К-средних RGB", title3="К-средних CIE Lab")

# min_samples = 150
# image_segmentation.compare_segmentations(image_segmentation.segment_image_dbscan,
#                                          min_samples=min_samples, target_color=color,
#                                          method_name="Метод DBSCAN", title1="Оригинальное изображение",
#                                          title2="DBSCAN RGB", title3="DBSCAN CIE Lab")

# seed_point, threshold = (100, 100), 170
# image_segmentation.compare_segmentations(image_segmentation.growing_seed_segmentation,
#                                          seed_point=seed_point, threshold=threshold,
#                                          method_name="Метод Growing seed", title1="Оригинальное изображение",
#                                          title2="Growing seed RGB", title3="Growing seed CIE Lab")

# threshold = 20
# image_segmentation.compare_segmentations(image_segmentation.region_growing, threshold=threshold,
#                                          method_name="Метод Region growing", title1="Оригинальное изображение",
#                                          title2="Region growing RGB", title3="Region growing CIE Lab")

# image_segmentation.compare_segmentations(image_segmentation.watershed_segmentation,
#                                          method_name="Метод Water Shed", title1="Оригинальное изображение",
#                                          title2="Water Shed RGB", title3="Water Shed CIE Lab")
