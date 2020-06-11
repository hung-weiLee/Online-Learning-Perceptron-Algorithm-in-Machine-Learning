import mnist_reader
import numpy as np
import matplotlib.pyplot as plt

# train_images, train_labels ==> 60000, 60000
X_train, y_train = mnist_reader.load_mnist('train')

# test_images, test_labels ==> 10000, 10000
X_test, y_test = mnist_reader.load_mnist('t10k')


##############################################


def predict(w_vector, j, str):
    W = []  # record every vector(0 ~ 9) => F(x, y)
    max = float("-inf")  # record the max value of w * every vector
    max_index = 0
    for i in range(0, 10):  # 10 classes
        ww = [0] * len(X_train[0]) * 10  # 7840
        if str == "train":
            ww[i * 784: (i * 784 + 784)] = X_train[j]
        else:  # test
            ww[i * 784: (i * 784 + 784)] = X_test[j]
        W.append(np.array(ww))  # vector
        d = np.dot(w_vector, np.array(ww))
        if d > max:
            max = d
            max_index = i
    return W, max_index


def c():
    w_sum = np.array([0] * len(X_train[0]) * 10)  # 7840
    w_vector = np.array([0] * len(X_train[0]) * 10)  # 7840
    Count = 1
    for i in range(0, 20):  # train 20 iteration
        for j in range(0, len(X_train)):  # 60000
            W, Yt = predict(w_vector, j, "train")  # predict
            # mistake
            if Yt != y_train[j]:
                w_vector = np.add(w_vector, np.subtract(W[y_train[j]], W[Yt]))
                w_sum = np.add(w_sum, (Count * np.subtract(W[y_train[j]], W[Yt])))
            Count += 1
        print(i + 1, "/", 20)

    w_avg = np.subtract(w_vector, w_sum / Count)  # average weight

    mistake = 0
    for k in range(0, len(X_test)):  # testing examples
        W, Yt = predict(w_avg, k, "test")  # predict
        # mistake
        if Yt != y_test[k]:
            mistake += 1

    accuracy = (len(X_test) - mistake) / len(X_test)
    print("average perceptron test accuracies :", accuracy)


##############################################


if __name__ == "__main__":
    c()
    #  plain perceptron test accuracies in Multi_Perceptron.py
