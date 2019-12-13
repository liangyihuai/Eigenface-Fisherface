import cv2
import os
import numpy as np
import pandas as pd
import sys


class ProfDataset:

    def __init__(self, path, rows=128, cols=128):
        self.path = path
        self.rows = rows
        self.cols = cols

    def get_data(self):
        """
        :return: (images, labels)
        """

        # list to hold all subject faces
        faces = []
        # list to hold labels for all subjects
        labels = []

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(self.path)

        # ------STEP-3--------
        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:
            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = self.path + "/" + image_name

            # read image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # add face to list of faces
            faces.append(image)
            # add label for this face
            splits = image_name.split('_', 2)
            label = splits[0] + '_' + splits[1]
            labels.append(label)

        return np.array(faces), np.array(labels)

    # mean face
    def mean_face(self, faces):
        mean_face = np.zeros((1, self.rows * self.cols))
        reshaped_faces = faces.reshape((faces.shape[0], self.rows * self.cols))
        for i in reshaped_faces:
            mean_face = np.add(mean_face, i)

        mean_face = np.divide(mean_face, float(len(faces))).flatten()
        return mean_face;


    def well_preprocessed_data(self):
        """
        the faces minus the mean face
        :return: tuple (faces, labels)
        """
        faces, labels = self.get_data()
        faces = np.reshape(faces, (faces.shape[0], self.rows * self.cols))
        faces = faces - self.mean_face(faces)
        return (faces, labels)