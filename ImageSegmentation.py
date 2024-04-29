import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster._hdbscan import hdbscan
from sklearn.preprocessing import StandardScaler
from numba import prange, njit


class ImageSegmentation:
    def __init__(self, image_path):
        self.RGB_image = None
        self.CIE_Lab_image = None
        self.__read_image(image_path)

    def __read_image(self, image_path: str):
        # Загружаем изображение
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.RGB_image = image
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        self.CIE_Lab_image = image_lab

    @staticmethod
    def create_one_plot(plot, num_image: int, image: np.ndarray, title: str):
        plot.subplot(1, 3, num_image)
        plot.imshow(image)
        # plot.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plot.title(title)
        plot.axis('off')

    def create_plot_images(self, original_image, image_segmented_rgb, image_segmented_lab, method_name="Segmentation",
                           title1='Original image', title2='RGB Segmented', title3='Lab Segmented'):
        plt.figure(figsize=(12, 4))

        # Первое изображение
        self.create_one_plot(plt, 1, original_image, title1)
        # Второе изображение
        self.create_one_plot(plt, 2, image_segmented_rgb, title2)
        # Третье изображение
        self.create_one_plot(plt, 3, image_segmented_lab, title3)

        plt.suptitle(method_name)
        plt.tight_layout()
        plt.show()

    def compare_segmentations(self, method, **kwargs):
        image = self.RGB_image
        image_lab = self.CIE_Lab_image

        segmented_image_rgb = method(image, kwargs)
        segmented_image_lab = method(image_lab, kwargs)

        self.create_plot_images(image, segmented_image_rgb, segmented_image_lab, method_name=kwargs["method_name"],
                                title1=kwargs["title1"], title2=kwargs["title2"], title3=kwargs["title3"])

    @staticmethod
    def segment_image_kmeans(image: np.ndarray, kwargs: dict):
        num_clusters = kwargs["num_clusters"]

        image_np = np.array(image)
        # Преобразуем изображение в одномерный массив
        image_reshaped = image_np.reshape((-1, 3)).astype(np.float32)

        # Определяем критерии останова для алгоритма kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        # Применяем метод kmeans
        _, labels, centers = (cv2.kmeans(data=image_reshaped, K=num_clusters, bestLabels=None,
                                         criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS))

        if "target_color" in kwargs.keys():
            target_color = kwargs["target_color"]
            # Найдем индекс центра кластера, наиболее близкого к целевому цвету
            target_center_index = np.argmin(np.linalg.norm(centers - target_color, axis=1))
            # Создаем сегментированное изображение, используя только выбранный центр кластера
            segmented_image_rgb = np.zeros_like(image_reshaped)
            for i in range(len(labels)):
                if labels[i] == target_center_index:
                    segmented_image_rgb[i] = centers[labels[i]]
        else:
            # Создаем сегментированное изображение, используя только выбранный центр кластера
            segmented_image_rgb = np.zeros_like(image_reshaped)
            for i in range(len(labels)):
                segmented_image_rgb[i] = centers[labels[i]]

        # Преобразуем метки кластеров обратно в форму изображения
        segmented_image_rgb = segmented_image_rgb.reshape(image_np.shape)

        return segmented_image_rgb.astype(np.uint8)

    @staticmethod
    def segment_image_dbscan(image: np.ndarray, kwargs: dict):
        min_samples = kwargs["min_samples"]
        image_np = np.array(image)
        image_reshaped = image_np.reshape((-1, 3)).astype(float)

        scaler = StandardScaler()
        image_scaled = scaler.fit_transform(image_reshaped)

        dbscan_rgb = hdbscan.HDBSCAN(min_cluster_size=min_samples)
        labels = dbscan_rgb.fit_predict(image_scaled)

        segmented_image = labels.reshape(image_np.shape[0], image_np.shape[1])
        return segmented_image.astype(np.uint8)

    @staticmethod
    def growing_seed_segmentation(image, kwargs):
        # Определяем функцию проверки соседних пикселей
        def check_neighbours(segmented_image, coordinate_current_seed, current_region_bright_mean,
                             current_depth, max_depth=990):
            if current_depth > max_depth:
                return
            # Получаем координаты семени и его значение
            seed_x, seed_y = coordinate_current_seed
            current_seed_value = image[seed_y, seed_x]  # Яркость.

            # Определяем соседние пиксели
            neighbours = [(seed_x + 1, seed_y), (seed_x - 1, seed_y), (seed_x, seed_y + 1), (seed_x, seed_y - 1)]
            # neighbours = [(seed_x + 1, seed_y), (seed_x - 1, seed_y), (seed_x, seed_y + 1), (seed_x, seed_y - 1),
            #               (seed_x + 1, seed_y + 1), (seed_x - 1, seed_y - 1),
            #               (seed_x + 1, seed_y - 1), (seed_x - 1, seed_y + 1)]

            # Перебираем соседние пиксели
            for i in range(len(neighbours)):
                neighbour_x, neighbour_y = neighbours[i]
                # Проверяем, находится ли соседний пиксель в пределах изображения
                if 0 <= neighbour_x < width and 0 <= neighbour_y < height:
                    neighbour_value = image[neighbour_y, neighbour_x]

                    # Если значение соседнего пикселя находится в пределах порогового значения, добавляем его к региону
                    if np.sum(np.abs(neighbour_value - current_region_bright_mean)) < region_threshold * 3:
                        segmented_image[neighbour_y, neighbour_x] = current_seed_value
                        # Обновляем среднее значение региона для текущего пикселя

                        # current_region_bright_mean = np.mean([current_region_bright_mean, neighbour_value])
                        check_neighbours(segmented_image, (neighbour_x, neighbour_y),
                                         current_region_bright_mean, current_depth + 1)

        seed_point, region_threshold = kwargs["seed_point"], kwargs["threshold"]
        # Копируем исходное изображение для работы
        segmented_image = np.zeros(image.shape)

        # Получаем размеры изображения
        height, width = image.shape[:2]
        # Выполняем выращивание семян для сегментации
        # ПРОГРАММА ПАДАЕТ С ОШИБКОЙ МАКСИМАЛЬНАЯ ГЛУБИНА РЕКУРСИВНОСТИ
        check_neighbours(segmented_image, seed_point, image[seed_point[1], seed_point[0]], 0)
        # segmented_image = np.int(segmented_image)
        return segmented_image.astype(np.uint8)
