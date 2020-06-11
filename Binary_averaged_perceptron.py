import mnist_reader
import numpy as np
import matplotlib.pyplot as plt

# train_images, train_labels ==> 60000, 60000
X_train, y_train = mnist_reader.load_mnist('train')

# test_images, test_labels ==> 10000, 10000
X_test, y_test = mnist_reader.load_mnist('t10k')

#  even labels: 0, 2, 4, 6, 8
#   odd labels: 1, 3, 5, 7, 9
dic = {0: 1, 2: 1, 4: 1, 6: 1, 8: 1, 1: -1, 3: -1, 5: -1, 7: -1, 9: -1}


##############################################


def c():
    w_sum = np.array([0] * len(X_train[0]))  # 784
    w_vector = np.array([0] * len(X_train[0]))  # 784
    Count = 1
    for i in range(0, 20):  # train 20 iteration
        for j in range(0, len(X_train)):  # 60000
            Yt = np.sign(np.dot(w_vector, X_train[j]))  # predict
            if Yt == 0: Yt = 1
            # mistake
            if Yt != dic[y_train[j]]:  # dic[y_train[j] = 1 or -1
                w_vector = np.add(w_vector, (np.dot(dic[y_train[j]], X_train[j])))
                w_sum = np.add(w_sum, np.dot(dic[y_train[j]] * Count, X_train[j]))
            Count += 1

    w_avg = np.subtract(w_vector, w_sum / Count)  # average weight

    mistake = 0
    for k in range(0, len(X_test)):  # testing examples
        Yt = np.sign(np.dot(w_avg, X_test[k]))
        if Yt == 0: Yt = 1
        # mistake
        if Yt != dic[y_test[k]]:
            mistake += 1

    accuracy = (len(X_test) - mistake) / len(X_test)
    print("average perceptron test accuracies :", accuracy)


##############################################


if __name__ == "__main__":
    c()
    #  plain perceptron test accuracies in Binary_Perceptron.py