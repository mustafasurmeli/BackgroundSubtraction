import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Videoları okuyun
opencv_video = cv2.VideoCapture('output_knn.mp4')
custom_video = cv2.VideoCapture('output.mp4')

mean_differences = []
iou_scores = []
ssim_scores = []

while True:
    # Kareleri oku
    ret1, opencv_frame = opencv_video.read()
    ret2, custom_frame = custom_video.read()

    if not ret1 or not ret2:  # Videoların sonuna geldiysek dur
        break

    # Gri tonlamaya çevir
    opencv_gray = cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2GRAY)
    custom_gray = cv2.cvtColor(custom_frame, cv2.COLOR_BGR2GRAY)

    # Görüntü boyutlarını eşitle
    custom_gray = cv2.resize(custom_gray, (opencv_gray.shape[1], opencv_gray.shape[0]))

    # Piksel farkı
    difference = cv2.absdiff(opencv_gray, custom_gray)
    mean_differences.append(np.mean(difference))

    # IoU
    _, opencv_binary = cv2.threshold(opencv_gray, 127, 1, cv2.THRESH_BINARY)
    _, custom_binary = cv2.threshold(custom_gray, 127, 1, cv2.THRESH_BINARY)
    intersection = np.logical_and(opencv_binary, custom_binary).sum()
    union = np.logical_or(opencv_binary, custom_binary).sum()
    iou_scores.append(intersection / union if union > 0 else 0)

    # SSIM
    ssim_score, _ = ssim(opencv_gray, custom_gray, full=True)
    ssim_scores.append(ssim_score)

# Ortalamaları hesapla
print(f"Mean Pixel Difference: {np.mean(mean_differences)}")
print(f"Mean IoU: {np.mean(iou_scores)}")
print(f"Mean SSIM: {np.mean(ssim_scores)}")
