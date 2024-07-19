import os

import cv2
import numpy as np

def load_images(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def create_labels(images1, images2):
    pairs = []
    labels = []

    for img1, img2 in zip(images1, images2):
        pairs.append([img1, img2])
        labels.append(1)

    # for i, img1 in enumerate(images1):
    #     idx = np.random.choice([x for x in range(len(images2)) if x != i])
    #     img2 = images2[idx]
    #     pairs.append([img1, img2])
    #     labels.append(0)
    #     print(i, idx)

    for i, img1 in enumerate(images1):
        exclude_indices = set(range(i - 11, i + 11))
        available_indices = [x for x in range(len(images2)) if x not in exclude_indices]
        idx = np.random.choice(available_indices)
        img2 = images2[idx]
        pairs.append([img1, img2])
        labels.append(0)
        print(i, idx)

    return np.array(pairs), np.array(labels)