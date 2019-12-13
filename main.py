from explore_faces94_dataset import Faces94
from fisher_face import FisherFace
from prof_dataset import ProfDataset
from sklearn.metrics import classification_report
import  numpy as np
from sklearn.decomposition import PCA

data_path_faces94 = 'images/faces94/'

# data_path_train = 'images/Training images'
data_path_test = 'images/Training images from professor_the second dataset/'



def get_data():
    faces94 = Faces94(data_path_faces94)
    X_train, y_train, X_test, y_test = faces94.get_images()

    # use this data as test data together with test data in faces94
    X_test2, y_test2 = ProfDataset(data_path_test).get_data()

    X_test2 = np.array([np.resize(i, (faces94.width, faces94.height)) for i in X_test2])
    X_test2 = X_test2.reshape((X_test2.shape[0], faces94.width * faces94.height))

    # the person in this test set is nobody, because they are on in training set
    NOBODY = '0'
    y_test2 = np.array([NOBODY for i in range(len(y_test2))])

    X_test = np.concatenate((X_test, X_test2))
    y_test = np.concatenate((y_test, y_test2))
    return faces94.X_train, faces94.y_train, X_test, y_test


if __name__ == '__main__':


    print('get data')
    X_train, y_train, X_test, y_test = get_data()

    print('fisher face')
    fisherFace = FisherFace(X_train, y_train, X_test, y_test)

    print('training')
    fisherFace.train(150, 100)

    print('accuracy')
    dist_threshold = 32.5931
    acc = fisherFace.accuracy(dist_threshold)
    print(acc)
    predicted_y = fisherFace.predict2(dist_threshold)
    print(classification_report(fisherFace.y_test, predicted_y))


    # print('explore best distance threshold')
    # distance_threshold = fisherFace.explore_best_distance_threshold(150, 100)
    # print(distance_threshold)

    #     32.5931





