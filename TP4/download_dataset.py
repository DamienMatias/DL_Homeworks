from threading import Thread
import threading
import numpy as np
import cv2
import os
from PIL import Image
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
import pickle
import time


class Download_imgs(Thread):
    def __init__(self, img_dir, img_label):
        Thread.__init__(self)
        self.img_dir = img_dir
        self.img_label = img_label

    def run(self):
        imgs = []
        labels = []
        with open(os.path.join(self.img_dir, "imagenet.synset.txt"), "r") as f:
            for url in f:
                img = self.download_and_resize_img(url)
                if img is not None:
                    labels.append(self.img_label)
                    labels.append(self.img_label)
                    imgs.append(img)
                    imgs.append(np.array(cv2.flip(img, 1)))

        data = np.asarray(imgs)
        targets = np.asarray(labels)

        try:
            f = open(os.path.join(self.img_dir, "dataset.pickle"), 'wb')
            save = {
                'data': data,
                'targets': targets
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', self.img_dir, ':', e)
            raise

        print(self.img_dir, " done")

    def download_and_resize_img(self, url):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img = np.array(img)
            if len(img.shape) < 3:
                #print(self.img_dir + "Grayscale; ignoring img")
                return None
            img = cv2.resize(img, (224, 224))
            return np.array(img)

        except:
            pass
            #print(self.img_dir + "Failed to load {}".format(url))


def download_from_root_dir(root_dir):
    subdir = next(os.walk(root_dir))[1]

    dict = {}
    threads = []
    label = 0

    for dir in subdir:
        object_name = dir.split("-")[1]
        dict[label] = object_name
        dir = os.path.join(root_dir, dir)

        threads.append(Download_imgs(dir, label))
        threads[label].start()
        label += 1

    while threading.activeCount() > 1:
        time.sleep(10)
        print(threading.activeCount())

    all_data = []
    for dir in subdir:
        dir = os.path.join(root_dir, dir)
        pickle_path = os.path.join(dir, "dataset.pickle")

        assert os.path.exists(pickle_path)
        with open(pickle_path, 'rb') as f:
            sub_data = pickle.load(f)
            all_data.append(sub_data)

    imgs = np.concatenate([data["data"] for data in all_data])
    labels = np.concatenate([data["targets"] for data in all_data])
    del all_data
    split_and_save(imgs, labels, dict, os.path.join(root_dir, "dataset.pickle"))

def split_and_save(data, labels, dict, path):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state =1)
    del data, labels
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    try:
        f = open(path, 'wb')
        save = {
            'train_dataset': X_train,
            'train_labels': y_train,
            'valid_dataset': X_val,
            'valid_labels': y_val,
            'test_dataset': X_test,
            'test_labels': y_test,
            'dict': dict
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', path, ':', e)
        raise


def main():
    download_from_root_dir("./image DB")

if __name__ == "__main__":
    main()
