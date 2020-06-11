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

accuracy_test = []  # calculate the accuracy for text examples


##############################################


def a():
    w_vector = np.array([0] * len(X_train[0]))  # 784
    mistake = [0] * 50
    for i in range(0, 50):  # train 50 iteration
        m_time = 0
        for j in range(0, len(X_train)):  # 60000
            Yt = np.sign(np.dot(w_vector, X_train[j]))  # predict
            if Yt == 0: Yt = 1
            # mistake
            if Yt != dic[y_train[j]]:  # dic[y_train[j] = 1 or -1
                m_time += 1
                w_vector = np.add(w_vector, (np.dot(dic[y_train[j]], X_train[j])))
        mistake[i] = m_time

    plt.plot([n for n in range(1, 51)], mistake)  # learning curve
    plt.title("Binary_Perceptron")
    plt.xlabel("50 iteration")
    plt.ylabel("mistake")
    plt.show()


##############################################


def test(wt_vector):  # testing examples
    mistake = 0
    for k in range(0, len(X_test)):  # 10000
        Yt = np.sign(np.dot(wt_vector, X_test[k]))
        if Yt == 0: Yt = 1
        # mistake
        if Yt != dic[y_test[k]]:
            mistake += 1

    accuracy_test.append((len(X_test) - mistake) / len(X_test))


def b():
    w_vector = np.array([0] * len(X_train[0]))  # 784
    accuracy = [0] * 20
    for i in range(0, 20):  # train 20 iteration
        m_time = 0
        for j in range(0, len(X_train)):  # 60000
            Yt = np.sign(np.dot(w_vector, X_train[j]))  # predict
            if Yt == 0: Yt = 1
            # mistake
            if Yt != dic[y_train[j]]:  # dic[y_train[j] = 1 or -1
                m_time += 1
                w_vector = np.add(w_vector, (np.dot(dic[y_train[j]], X_train[j])))
        wt_vector = np.copy(w_vector)
        test(wt_vector)  # call text function
        accuracy[i] = (len(X_train) - m_time) / len(X_train)

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