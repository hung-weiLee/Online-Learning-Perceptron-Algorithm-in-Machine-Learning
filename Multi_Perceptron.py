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

def a():
    w_vector = np.array([0] * len(X_train[0]) * 10)  # 7840
    mistake = [0] * 50
    for i in range(0, 50):  # train 50 iteration
        m_time = 0
        for j in range(0, len(X_train)):  # 60000
            W, Yt = predict(w_vector, j, "train")  # predict
            # mistake
            if Yt != y_train[j]:
                m_time += 1
                w_vector = np.add(w_vector, np.subtract(W[y_train[j]], W[Yt]))
        mistake[i] = m_time
        print(i + 1, "/", 50)

    plt.plot([n for n in range(1, 51)], mistake)  # learning curve
    plt.title("Multi_Perceptron")
    plt.xlabel("50 iteration")
    plt.ylabel("mistake")
    plt.show()


##############################################


def test(wt_vector):  # testing examples
    mistake = 0
    for k in range(0, len(X_test)):  # 10000
        W, Yt = predict(wt_vector, k, "test")  # predict
        # mistake
        if Yt != y_test[k]:
            mistake += 1

    accuracy_test.append((len(X_test) - mistake) / len(X_test))


def b():
    w_vector = np.array([0] * len(X_train[0]) * 10)  # 7840
    accuracy = [0] * 20  # train 20 iteration
    for i in range(0, 20):
        m_time = 0
        for j in range(0, len(X_train)):  # 60000
            W, Yt = predict(w_vector, j, "train")  # predict
            # mistake
            if Yt != y_train[j]:
                m_time += 1
                w_vector = np.add(w_vector, np.subtract(W[y_train[j]], W[Yt]))
        wt_vector = np.copy(w_vector)
        test(wt_vector)  # call text function
        accuracy[i] = (len(X_train) - m_time) / len(X_train)
        print(i + 1, "/", 20)

    plt.plot([n for n in range(1, 21)], accuracy)  # accuracy curve
    plt.title("accuracy curve for train (Perceptron)")
    plt.xlabel("20 iteration")
    plt.ylabel("accuracy")
    plt.show()

    plt.plot([n for n in range(1, 21)], accuracy_test)  # accuracy curve
    plt.title("accuracy curve for test (Perceptron)")
    plt.xlabel("20 iteration")
    plt.ylabel("accuracy")
    plt.show()


##############################################


if __name__ == "__main__":
    a()
    b()
    # print("plain perceptron test accuracies :", accuracy_test.pop())