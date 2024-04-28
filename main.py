import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def segment_image_kmeans(image, num_clusters):
    image_np = np.array(image)
    # Преобразуем изображение в одномерный массив
    image_reshaped = image_np.reshape((-1, 3)).astype(np.float32)

    # Определяем критерии останова для алгоритма kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Применяем метод kmeans
    _, labels, centers = cv2.kmeans(image_reshaped, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Восстановление сегментированного изображения
    segmented_image_rgb = np.zeros_like(image_reshaped)
    for i in range(len(labels)):
        segmented_image_rgb[i] = centers[labels[i]]

    # Преобразуем метки кластеров обратно в форму изображения
    segmented_image_rgb = segmented_image_rgb.reshape(image_np.shape)

    return segmented_image_rgb.astype(np.uint8)


def segment_image_dbscan(image, eps, min_samples):
    # Преобразуем изображение в массив numpy
    image_np = np.array(image)
    # Преобразуем изображение в формат RGB
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # Преобразуем изображение в одномерный массив для DBSCAN
    image_reshaped = image_rgb.reshape((-1, 3))
    # Масштабируем признаки
    scaler = StandardScaler()
    image_scaled = scaler.fit_transform(image_reshaped.astype(float))
    # Применяем метод DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(image_scaled)
    # Получаем метки кластеров для каждого пикселя
    labels = dbscan.labels_
    # Преобразуем метки кластеров обратно в форму изображения
    segmented_image = labels.reshape(image_rgb.shape[0], image_rgb.shape[1])
    return segmented_image


def compare_segmentations(image_path, num_clusters, eps, min_samples):
    # Загружаем изображение
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Преобразуем изображение в цветовое пространство CIE Lab
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Преобразование изображения в формат RGB
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)

    cv2.imshow("RGB image 1", image.astype(np.uint8))
    segmented_image_rgb_kmeans = segment_image_kmeans(image, num_clusters)
    cv2.imshow("RGB image 2", image_lab.astype(np.uint8))
    segmented_image_lab_kmeans = segment_image_kmeans(image_lab, num_clusters)

    segmented_image_rgb_dbscan = np.zeros(image.shape)

    segmented_image_lab_dbscan = np.zeros(image.shape)

    # # Сегментация по цветовому пространству RGB с использованием метода DBSCAN
    # segmented_image_rgb_dbscan = segment_image_dbscan(image, eps, min_samples)
    #
    # # Сегментация по цветовому пространству CIE Lab с использованием метода DBSCAN
    # segmented_image_lab_dbscan = segment_image_dbscan(image_lab, eps, min_samples)

    return segmented_image_rgb_kmeans, segmented_image_lab_kmeans, \
        segmented_image_rgb_dbscan, segmented_image_lab_dbscan


# Указываем путь к изображению
image_path = "321.jpg"

# Задаем параметры для методов кластеризации
num_clusters = 4
eps = 20
min_samples = 50

# Получаем результаты сегментации для всех четырех вариантов
segmented_image_rgb_kmeans, segmented_image_lab_kmeans, segmented_image_rgb_dbscan, segmented_image_lab_dbscan = \
    compare_segmentations(image_path, num_clusters, eps, min_samples)

# Выводим результаты сегментации
cv2.imshow("RGB + kmeans", segmented_image_rgb_kmeans)
cv2.imshow("Lab + kmeans", segmented_image_lab_kmeans)
# cv2.imshow("RGB + DBSCAN", segmented_image_rgb_dbscan)
# cv2.imshow("Lab + DBSCAN", segmented_image_lab_dbscan)
cv2.waitKey(0)
cv2.destroyAllWindows()
