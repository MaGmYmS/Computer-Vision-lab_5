from ImageSegmentation import ImageSegmentation

# Указываем путь к изображению
# image_path = "images/321.jpg"
# image_path = "images/123.webp"
image_path = "images/flower.jpg"
image_segmentation = ImageSegmentation(image_path)
color = image_segmentation.RGB_image[100, 100]
num_clusters = 12
image_segmentation.compare_segmentations(image_segmentation.segment_image_kmeans,
                                         num_clusters=num_clusters, target_color=color,
                                         method_name="Метод k-means", title1="Оригинальное изображение",
                                         title2="К-средних RGB", title3="К-средних CIE Lab")

# min_samples = 150
# image_segmentation.compare_segmentations(image_segmentation.segment_image_dbscan, min_samples=min_samples,
#                                          method_name="Метод DBSCAN", title1="Оригинальное изображение",
#                                          title2="DBSCAN RGB", title3="DBSCAN CIE Lab")
#
# seed_point, threshold = (100, 100), 5
# image_segmentation.compare_segmentations(image_segmentation.growing_seed_segmentation,
#                                          seed_point=seed_point, threshold=threshold,
#                                          method_name="Метод Region growing", title1="Оригинальное изображение",
#                                          title2="Region growing RGB", title3="Region growing CIE Lab")
