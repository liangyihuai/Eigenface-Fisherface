import cv2
import os
import numpy as np
import pandas as pd
import sys

from matplotlib import pyplot as plt

from sklearn.model_selection import cross_validate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

class Faces94:
    def __init__(self, path='images/faces94/', WIDTH=200, HEIGHT=180):
        self.path = path
        self.width = WIDTH
        self.height = HEIGHT

    def __get_train_test_image_path(self, imageType='.jpg'):
        train_paths = []  #
        test_paths = []

        person_names = os.listdir(self.path)  # each person's images in a folder
        all_image_names = []
        for person_name in person_names:
            sub_path = self.path + person_name + "/";
            one_person_image_names = os.listdir(sub_path)
            one_person_image_names = [sub_path + i for i in one_person_image_names if i.endswith(imageType)]
            if len(one_person_image_names) > 3:
                train_paths += one_person_image_names[:-2]
                test_paths += one_person_image_names[-2:]  # use the last two images of each person as test data

        return np.array(train_paths), np.array(test_paths)

    def __get_images(self, paths):
        images = []
        labels = []

        # read image
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            images.append(img)
            labels.append(path.rsplit('/', 1)[-1].split('.', 1)[0])
        images = np.array(images)
        images = images.reshape((images.shape[0], self.width * self.height))

        return images, labels

    def get_images(self):
        """

        :return: (X_train, y_train, X_test, y_test)
        """
        if not hasattr(self, 'X_train'):
            train_paths, test_paths = self.__get_train_test_image_path()
            self.X_train, self.y_train = self.__get_images(train_paths)
            self.X_test, self.y_test = self.__get_images(test_paths)
        return self.X_train, self.y_train, self.X_test, self.y_test




