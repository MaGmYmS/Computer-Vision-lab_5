import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster._hdbscan import hdbscan
from sklearn.preprocessing import StandardScaler
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.segmentation import flood
from tqdm import tqdm


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
    def __create_one_plot(plot, num_image: int, image: np.ndarray, title: str):
        plot.subplot(1, 3, num_image)
        plot.imshow(image)
        # plot.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plot.title(title)
        plot.axis('off')

    def __create_plot_images(self, original_image, image_segmented_rgb, image_segmented_lab, method_name="Segmentation",
                             title1='Original image', title2='RGB Segmented', title3='Lab Segmented'):
        plt.figure(figsize=(12, 4))

        # Первое изображение
        self.__create_one_plot(plt, 1, original_image, title1)
        # Второе изображение
        self.__create_one_plot(plt, 2, image_segmented_rgb, title2)
        # Третье изображение
        self.__create_one_plot(plt, 3, image_segmented_lab, title3)

        plt.suptitle(method_name)
        plt.tight_layout()
        plt.show()

    def compare_segmentations(self, method, **kwargs):
        image = self.RGB_image
        image_lab = self.CIE_Lab_image

        segmented_image_rgb = method(image, kwargs)
        segmented_image_lab = method(image_lab, kwargs)

        self.__create_plot_images(image, segmented_image_rgb, segmented_image_lab, method_name=kwargs["method_name"],
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
            target_color = kwargs["target_color"][0]
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

        dbscan = hdbscan.HDBSCAN(min_cluster_size=min_samples)
        labels = dbscan.fit_predict(image_scaled)

        if "target_color" in kwargs.keys():
            target_color, x_target_color, y_target_color = kwargs["target_color"]
            # Находим индекс кластера, наиболее близкого к целевому цвету
            # centers = np.array([np.mean(image_scaled[labels == i], axis=0) for i in range(len(set(labels)))])
            target_cluster_index = labels[x_target_color * y_target_color]
            # Создаем сегментированное изображение, используя только пиксели из выбранного кластера
            segmented_image = np.zeros_like(labels)
            for i in range(len(labels)):
                if labels[i] == target_cluster_index:
                    segmented_image[i] = labels[i]
        else:
            segmented_image = labels

        # Преобразуем метки кластеров обратно в форму изображения
        segmented_image = segmented_image.reshape(image_np.shape[0], image_np.shape[1])

        return segmented_image.astype(np.uint8)

    @staticmethod
    def growing_seed_segmentation(image: np.ndarray, kwargs: dict):
        seed_point, region_threshold = kwargs["seed_point"], kwargs["threshold"]
        segmented_image = np.copy(image)
        height, width = image.shape[:2]

        stack = [seed_point]  # Используем стек для хранения пикселей, которые нужно проверить
        visited = set()  # Множество для отслеживания уже посещенных пикселей

        while stack:
            current_seed = stack.pop()  # Берем текущий пиксель из стека
            seed_x, seed_y = current_seed
            current_seed_value = image[seed_y, seed_x]

            # Проверяем, был ли этот пиксель уже посещен
            if current_seed in visited:
                continue

            # Добавляем текущий пиксель в посещенные
            visited.add(current_seed)

            # Обновляем значение сегментированного изображения
            segmented_image[seed_y, seed_x] = [0, 0, 0]

            # Определяем соседние пиксели
            neighbours = [(seed_x + 1, seed_y), (seed_x - 1, seed_y),
                          (seed_x, seed_y + 1), (seed_x, seed_y - 1),
                          (seed_x + 1, seed_y + 1), (seed_x + 1, seed_y - 1),
                          (seed_x - 1, seed_y - 1), (seed_x - 1, seed_y + 1)]

            # Перебираем соседние пиксели
            for neighbour_x, neighbour_y in neighbours:
                # Проверяем, находится ли соседний пиксель в пределах изображения
                if 0 <= neighbour_x < width and 0 <= neighbour_y < height:
                    neighbour_value = image[neighbour_y, neighbour_x]

                    # Проверяем, был ли соседний пиксель уже посещен
                    if (neighbour_x, neighbour_y) in visited:
                        continue

                    # Если значение соседнего пикселя находится в пределах порогового значения, добавляем его в стек
                    # print(np.sum(np.abs(neighbour_value - current_seed_value)))
                    if np.sum(np.abs(neighbour_value - current_seed_value)) < region_threshold * 3:
                        stack.append((neighbour_x, neighbour_y))

        return segmented_image.astype(np.uint8)

    @staticmethod
    def region_growing(image: np.ndarray, kwargs: dict):
        threshold = kwargs.get("threshold", 10)

        # Преобразуем изображение в оттенки серого
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Создаем пустое изображение для хранения сегментированной области
        segmented_image = np.zeros_like(gray_image)
        height, width = gray_image.shape

        # Инициализируем массив для отслеживания принадлежности пикселей к регионам
        region_membership = np.zeros_like(gray_image)

        # Создаем список для хранения областей (регионов)
        regions = []

        # Перебираем каждый пиксель изображения
        for y in tqdm(range(height)):
            for x in range(width):
                # Если пиксель уже относится к какому-то региону, пропускаем его
                if region_membership[y, x] != 0:
                    continue
                if len(regions) == 0:
                    new_region = [(x, y)]
                    regions.append(new_region)
                    continue
                # Получаем яркость текущего пикселя
                pixel_intensity = gray_image[y, x]
                intensity_regions = [[gray_image[j, i] for i, j in region] for region in regions]
                avg_regions = [np.mean(intensity_region) for intensity_region in intensity_regions]
                if (abs(avg_regions - pixel_intensity) > threshold).all():  # Если разность интенсивности и всех
                    # регионов больше границы, то создаем новый регион и добавляем в него текущий пиксель.
                    new_region = [(x, y)]
                    regions.append(new_region)
                    region_membership[y, x] = len(regions) + 1
                elif (abs(pixel_intensity - avg_regions) <= threshold).any():  # Если разность интенсивности и хотя бы
                    # одного региона меньше или равно границы, то создаем новый регион и
                    # добавляем в него текущий пиксель.
                    new_region = [(x, y)]
                    regions.append(new_region)
                    region_membership[y, x] = len(regions) + 1
                else:
                    for i in range(len(regions)):
                        for j in range(len(regions)):
                            avg_region_i = np.mean([gray_image[x_coord, y_coord] for x_coord, y_coord in regions[i]])
                            avg_region_j = np.mean([gray_image[x_coord, y_coord] for x_coord, y_coord in regions[j]])
                            if abs(avg_region_i - avg_region_j <= threshold):
                                # Если разность средних яркостей двух областей меньше порога, области сливаем воедино
                                regions[i] = regions[i] + regions[j]
                                regions.remove(regions[j])
                            else:
                                # Иначе добавляем пиксель А к региону с наименьшим отклонением
                                if abs(avg_region_i - pixel_intensity) > abs(avg_region_j - pixel_intensity):
                                    regions[j].append((x, y))
                                else:
                                    regions[i].append((x, y))

            # Заполняем сегментированное изображение в соответствии с регионами
        for region_id, region in enumerate(regions):
            for x, y in region:
                segmented_image[y, x] = region_id + 1

        return segmented_image

    @staticmethod
    def watershed_segmentation(image: np.ndarray, kwargs: dict):
        # Вычисляем градиент изображения
        gradient = ndimage.morphological_gradient(image, size=3)

        # Находим локальные максимумы градиента
        local_maxi = peak_local_max(gradient, labels=image)

        # Применяем метод водораздела
        markers = ndimage.label(local_maxi)[0]
        segmented_image = watershed(-gradient, markers, mask=image)

        return segmented_image
