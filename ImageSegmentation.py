import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster._hdbscan import hdbscan
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Pool


class ImageSegmentation:
    def __init__(self, image_path):
        self.RGB_image = None
        self.CIE_Lab_image = None
        self.__read_image(image_path)

    def __read_image(self, image_path: str):
        # Загружаем изображение
        parts = image_path.split('/')
        parts = parts[len(parts) - 2:]
        image_path = '/'.join(parts)
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

        start_time = time.time()
        segmented_image_rgb = method(image, kwargs)
        end_time = time.time() - start_time
        print(kwargs["title2"], "отработал за", round(end_time / 60, 2))
        segmented_image_lab = segmented_image_rgb
        # start_time = time.time()
        # segmented_image_lab = method(image_lab, kwargs)
        # end_time = time.time() - start_time
        # print(kwargs["title3"], "отработал за", round(end_time / 60, 2))

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

        if kwargs["target_color"] is not None:
            target_color, _, _ = kwargs["target_color"][0]
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

        if kwargs["target_color"] is not None:
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

    def region_growing(self, image: np.ndarray, kwargs: dict):
        threshold = kwargs.get("threshold", 10)
        color = kwargs.get("color", None)
        if color is not None:
            color = [[color[1:]]]
        else:
            color = (0, 0)
        # Преобразуем изображение в оттенки серого
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return self.__region_growing(gray_image, threshold, color)

    def __check_merge_region(self, gray_image, regions, threshold):
        # Получаем средние интенсивности для всех регионов одновременно
        avg_regions = self.__calculate_average_regions(gray_image, regions)

        # Создаем матрицу расстояний между средними интенсивностями
        diff_matrix = np.abs(avg_regions[:, None] - avg_regions)
        np.fill_diagonal(diff_matrix, np.inf)  # Исключаем сравнение региона с самим собой

        # Находим индексы пар регионов, которые можно объединить
        merge_ri_rj = np.transpose(np.where(diff_matrix <= threshold))

        return merge_ri_rj.tolist()

    def __merge_region(self, indexes_merge_regions, regions):
        # Получаем индексы регионов для слияния
        merge_indices = np.array(indexes_merge_regions)

        # Создаем новый регион, в который будем все сливать
        new_merge_region = [regions[merge_indices[0, 0]]]

        # Идем по циклу, добавляем регионы в массив для слияния
        for i in range(1, len(merge_indices)):
            new_merge_region.extend(regions[merge_indices[i, 1]])

        # Удаляем регионы
        regions = [region for idx, region in enumerate(regions) if idx not in merge_indices[:, 1]]

        # Добавляем результат в наш массив с регионами
        regions.append(new_merge_region)

        return regions

    @staticmethod
    def __calculate_average_regions(gray_image, regions):
        avg_regions = []
        for region in regions:
            # Преобразуем координаты региона в массив numpy
            region_array = np.array(region)
            # Извлекаем значения пикселей из изображения с помощью массива координат региона
            region_values = gray_image[region_array[:, 1], region_array[:, 0]]
            # Вычисляем среднее значение для региона и добавляем его в список
            avg_regions.append(np.mean(region_values))
        return np.array(avg_regions)

    def __region_growing(self, gray_image: np.ndarray, threshold: int, color_coordinate: (int, int) = None):
        # Создаем пустое изображение для хранения сегментированной области
        segmented_image = np.zeros_like(gray_image)
        height, width = gray_image.shape

        # Создаем список для хранения областей (регионов)
        regions = [[color_coordinate]]

        # Вычисляем среднюю интенсивность по всем регионам
        avg_regions = self.__calculate_average_regions(gray_image, regions)

        # Перебираем каждый пиксель изображения
        for y in tqdm(range(height), desc="Region growing"):
            for x in range(width):
                if (x, y) == color_coordinate:
                    continue
                # Получаем яркость текущего пикселя
                pixel_intensity = gray_image[y, x]

                # Вычисляем разность интенсивности текущего пикселя и средней интенсивности каждого региона
                diff = np.abs(avg_regions - pixel_intensity)

                # Если разность больше порога, создаем новый регион
                if (diff > threshold).all():
                    new_region = [(x, y)]
                    regions.append(new_region)
                    avg_regions = np.concatenate((avg_regions, [pixel_intensity]))
                else:
                    # Находим индекс региона с минимальной разностью и добавляем текущий пиксель в этот регион
                    min_diff_index = np.argmin(diff)
                    regions[min_diff_index].append((x, y))
                    # Обновляем среднюю интенсивность региона
                    avg_regions[min_diff_index] = (
                        np.mean(np.array([gray_image[j, i] for i, j in regions[min_diff_index]])))

        # Заполняем сегментированное изображение в соответствии с регионами
        print("Количество регионов:", len(regions))
        for region_id, region in enumerate(regions):
            for x, y in region:
                segmented_image[y, x] = region_id + 1

        return segmented_image.astype(np.uint8)

    @staticmethod
    def watershed_segmentation(image: np.ndarray, kwargs: dict):
        # Преобразование изображения в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Применение пороговой обработки Оцу для получения двоичного изображения
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Удаление шумов с помощью морфологического замыкания
        kernel = np.ones((2, 2), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Определение области "уверенного фона" путем дилатации
        sure_bg = cv2.dilate(closing, kernel, iterations=3)

        # Поиск области "уверенного переднего плана" с использованием преобразования расстояний
        dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)

        # Пороговая обработка преобразованного расстояния
        ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

        # Поиск неизвестной области
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Маркировка компонентов
        ret, markers = cv2.connectedComponents(sure_fg)

        # Добавление единицы ко всем меткам, чтобы избежать метки фона равной 0
        markers = markers + 1

        # Маркировка неизвестной области нулем
        markers[unknown == 255] = 0

        # Применение алгоритма водораздела
        markers = cv2.watershed(image, markers)

        # Создание цветовой карты на основе оттенка в цветовом пространстве HSV
        hsv_colors = np.zeros((markers.max() + 1, 3), dtype=np.uint8)
        for i in range(1, markers.max() + 1):
            hsv_colors[i] = [int(180 * i / (markers.max() + 1)), 255, 255]

        # Преобразование HSV в BGR для OpenCV
        bgr_colors = cv2.cvtColor(hsv_colors.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)

        # Окрашивание сегментированных областей
        segmented_image = bgr_colors[markers]

        return segmented_image.astype(np.uint8)
