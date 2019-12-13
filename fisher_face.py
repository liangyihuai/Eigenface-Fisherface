import cv2
import os
import numpy as np
import pandas as pd
import sys
import time

from matplotlib import pyplot as plt

from sklearn.model_selection import cross_validate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

class FisherFace:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.NOBODY = '0';  # mean the test face is nobody that the system does not know.
        self.N_train = len(X_train)

    # L2 norm distance
    def __distance(self, arr1, arr2):
        sub = np.subtract(arr1, arr2)
        return np.sqrt(np.dot(sub, sub))

    # test faces
    # params: distance_threshold, if a test face has a big distance with all training face,
    # which is greater than distance_threshold, it can cosider there is no such person in the training faces.
    def predict(self, new_space_test_faces, new_space_train_faces, distance_threshold=sys.maxsize):
        predict_result = []  # store tuple (training label, training face, min distance)
        size = len(new_space_train_faces)
        for test_face in new_space_test_faces:

            min_distance = sys.maxsize
            predicted_label = None
            # get the training label and training face of the min distance
            for i in range(size):
                # calculate distance
                sub = np.subtract(new_space_train_faces[i], test_face)
                dis = np.sqrt(np.dot(sub, sub))

                # dis = self.__distance(new_space_train_faces[i], test_face)
                if dis < min_distance:
                    min_distance = dis
                    predicted_label = self.y_train[i]
            if (min_distance < distance_threshold):
                predict_result.append((predicted_label, min_distance))
            else:
                predict_result.append((self.NOBODY, min_distance))
        return predict_result

    def predict2(self, distance_threshold=sys.maxsize):
        predict_result = self.predict(self.test_transformed, self.train_transformed, distance_threshold)
        predicted_Y = [predict_label for (predict_label, _) in predict_result]
        return predicted_Y


    def display_distance_info(self, new_space_test_faces, new_space_train_faces):
        predict_result = self.predict(new_space_test_faces, new_space_train_faces)

        sum_correct_dists = 0.0;
        sum_all_dists = 0.0
        count = 0
        max_correct_dist = 0.0
        max_dist = 0.0
        min_dist = sys.maxsize
        for i in range(len(self.X_test)):
            predicted_label, predicted_face, dist = predict_result[i]

            sum_all_dists += dist
            if dist > max_dist:
                max_dist = dist;
            if dist < min_dist:
                min_dist = dist;
            if self.y_test[i] == predicted_label:
                sum_correct_dists += dist
                count += 1
                if dist > max_correct_dist: max_correct_dist = dist;

        ave_correct_dist = sum_correct_dists / count
        ave_all_dist = sum_all_dists / len(self.X_test)
        print("average distance of correct prediction", ave_correct_dist)
        print("average distance of all prediction", ave_all_dist)
        print('max correct distance', max_correct_dist)
        print('min distance', min_dist)
        print('max distance', max_dist)

    # whether 'dataset' contains 'a_data'
    def __contains(self, dataset, a_data):
        for v in dataset:
            if v == a_data:
                return True;
        return False

    def accuracy(self, distance_threshold):
        if not hasattr(self, 'train_transformed'):
            print('train model first')
            return ;
        predict_result = self.predict(self.test_transformed, self.train_transformed, distance_threshold)
        predicted_Y = [predict_label for (predict_label, _) in predict_result]
        return np.mean(self.y_test == predicted_Y)

    def train(self, n_component_pca, n_component_lda):
        pca = PCA(n_components=n_component_pca)
        pca.fit(self.X_train)

        lda = LinearDiscriminantAnalysis(n_components=n_component_lda)
        self.train_transformed = lda.fit_transform(pca.transform(self.X_train), self.y_train)
        self.test_transformed = lda.transform(pca.transform(self.X_test))

        # pca.explained_variance_ratio_

    def __max_correct_distance(self, n_component_pca, n_component_lda):
        if not hasattr(self, "train_transformed"):
            self.train(n_component_pca, n_component_lda)
        predict_result = self.predict(self.test_transformed, self.train_transformed)

        max_correct_dist = 0.0

        for i in range(len(self.X_test)):
            predicted_label, dist = predict_result[i]

            if self.y_test[i] == predicted_label:
                if dist > max_correct_dist:
                    max_correct_dist = dist;
        return max_correct_dist

    def explore_best_distance_threshold(self, n_component_pca, n_component_lda):
        max_correct_dist = self.__max_correct_distance(n_component_pca, n_component_lda)

        best_acc = 0
        min_thres = max_correct_dist / 2
        max_thres = max_correct_dist
        NUM_RANDOM_THRESHOLD = 1000
        best_distance_thres = 0

        ran_arr = np.random.rand(NUM_RANDOM_THRESHOLD) * (max_thres - min_thres) + min_thres
        for thres in ran_arr:
            start = time.time()
            predict_result = self.predict(self.test_transformed, self.train_transformed, thres)
            predicted_Y = [predict_label for (predict_label, _) in predict_result]
            acc = np.mean(self.y_test == predicted_Y)

            if acc > best_acc:
                best_distance_thres = thres
                best_acc = acc

            end = time.time()
            print(end - start)

        return best_distance_thres

