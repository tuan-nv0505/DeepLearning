from pathlib import Path
import sys
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
from collections import defaultdict
import cv2 #type:ignore

def read_data(folder_path: Path):
    data = defaultdict(lambda: np.array([]))
    for file in folder_path.iterdir():
        if file.is_file() and file.name.endswith('.npy'):
            data[file.name[:-4]] = np.load(file)
    return data

def predict(img_test, matrix_weight):
    return np.argmax(matrix_weight @ img_test.reshape((-1, 1)))

if __name__ == '__main__':
    res = read_data(PROJECT_DIR / 'datasets/quick_draw')
    data = list(res.values())
    label = list(res.keys())

    train = [x[:-10, :] for x in data]
    test = np.array([x[-10:, :] for x in data]).reshape(-1, 784)

    w = np.empty((0, 784))
    for x in train:
        w = np.vstack((w, np.mean(x, axis=0)))


    img_test = test[np.random.randint(0, test.shape[0])]
    print(label[predict(img_test, w)])
    cv2.imshow('test', img_test.reshape((28, 28)))
    cv2.waitKey(0)

