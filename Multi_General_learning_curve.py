import mnist_reader
import numpy as np
import matplotlib.pyplot as plt

# train_images, train_labels ==> 60000, 60000
X_train, y_train = mnist_reader.load_mnist('train')

# test_images, test_labels ==> 10000, 10000
X_test, y_test = mnist_reader.load_mnist('t10k')

accuracy_test = []  # calculate the accuracy for text examples


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


def test(wt_vector):  # testing examples
    mistake = 0
    for k in range(0, len(X_test)):  # 10000
        W, Yt = predict(wt_vector, k, "test")  # predict
        # mistake
        if Yt != y_test[k]:
            mistake += 1

    accuracy_test.append((len(X_test) - mistake) / len(X_test))


def d_Perceptron():
    w_vector = np.array([0] * len(X_train[0]) * 10)  # 7840
    for i in range(0, 20):  # train 20 iteration
        for j in range(0, len(X_train)):  # 60000
            W, Yt = predict(w_vector, j, "train")  # predict
            # mistake
            if Yt != y_train[j]:
                w_vector = np.add(w_vector, np.subtract(W[y_train[j]], W[Yt]))
            if (j + 1) % 5000 == 0:  # 5000 training examples
                wt_vector = np.copy(w_vector)
                test(wt_vector)  # call test function
        print(i + 1, "/", 20)

    plt.plot([n for n in range(1, 1200001, 5000)], accuracy_test)  # general learning curve
    plt.title("general learning curve (Multi_Perceptron)")
    plt.xlabel("training examples")
    plt.ylabel("testing accuracy")
    plt.show()
    # print("d_Perceptron: ", accuracy_test)


def d_PA():
    for e in range(0, len(accuracy_test)):
        accuracy_test.pop()  # clean the list

    w_vector = np.array([0] * len(X_train[0]) * 10)  # 7840
    for i in range(0, 20):  # train 20 iteration
        for j in range(0, len(X_train)):  # 60000
            W, Yt = predict(w_vector, j, "train")  # predict
            # mistake
            if Yt != y_train[j]:
                T = (1 - np.subtract(np.dot(w_vector, W[y_train[j]]), np.dot(w_vector, W[Yt]))) / np.linalg.norm(np.subtract(W[y_train[j]], W[Yt])) ** 2
                w_vector = np.add(w_vector, np.dot(T, np.subtract(W[y_train[j]], W[Yt])))
            if (j + 1) % 5000 == 0:  # 5000 training examples
                wt_vector = np.copy(w_vector)
                test(wt_vector)  # call test function
        print(i + 1, "/", 20)

    plt.plot([n for n in range(1, 1200001, 5000)], accuracy_test)  # general learning curve
    plt.title("general learning curve (Multi_PA)")
    plt.xlabel("training examples")
    plt.ylabel("testing accuracy")
    plt.show()
    # print("d_PA: ", accuracy_test)


##############################################


if __name__ == "__main__":
    d_Perceptron()
    d_PA()