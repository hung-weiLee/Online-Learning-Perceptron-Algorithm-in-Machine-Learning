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


def test(wt_vector):  # testing examples
    mistake = 0
    for k in range(0, len(X_test)):  # 10000
        Yt = np.sign(np.dot(wt_vector, X_test[k]))
        if Yt == 0: Yt = 1
        # mistake
        if Yt != dic[y_test[k]]:
            mistake += 1

    accuracy_test.append((len(X_test) - mistake) / len(X_test))


def d_Perceptron():
    w_vector = np.array([0] * len(X_train[0]))  # 784
    for i in range(0, 20):  # train 20 iteration
        for j in range(0, len(X_train)):  # 60000
            Yt = np.sign(np.dot(w_vector, X_train[j]))  # predict
            if Yt == 0: Yt = 1
            # mistake
            if Yt != dic[y_train[j]]:  # dic[y_train[j] = 1 or -1
                w_vector = np.add(w_vector, (np.dot(dic[y_train[j]], X_train[j])))
            if (j + 1) % 5000 == 0:  # 5000 training examples
                wt_vector = np.copy(w_vector)
                test(wt_vector)  # call test function

    plt.plot([n for n in range(1, 1200001, 5000)], accuracy_test)  # general learning curve
    plt.title("general learning curve (Binary_Perceptron)")
    plt.xlabel("training examples")
    plt.ylabel("testing accuracy")
    plt.show()
    # print("d_Perceptron: ", accuracy_test)


def d_PA():
    for e in range(0, len(accuracy_test)):
        accuracy_test.pop()  # clean the list

    w_vector = np.array([0] * len(X_train[0]))  # 784
    for i in range(0, 20):  # train 20 iteration
        for j in range(0, len(X_train)):  # 60000
            Yt = np.sign(np.dot(w_vector, X_train[j]))  # predict
            if Yt == 0: Yt = 1
            # mistake
            if Yt != dic[y_train[j]]:  # dic[y_train[j] = 1 or -1
                T = (1 - np.dot(dic[y_train[j]], np.dot(w_vector, X_train[j]))) / np.linalg.norm(X_train[j]) ** 2
                w_vector = np.add(w_vector, np.dot(T, (np.dot(dic[y_train[j]], X_train[j]))))
            if (j + 1) % 5000 == 0:  # 5000 training examples
                wt_vector = np.copy(w_vector)
                test(wt_vector)  # call test function

    plt.plot([n for n in range(1, 1200001, 5000)], accuracy_test)  # general learning curve
    plt.title("general learning curve (Binary_PA)")
    plt.xlabel("training examples")
    plt.ylabel("testing accuracy")
    plt.show()
    # print("d_PA: ", accuracy_test)


##############################################


if __name__ == "__main__":
    d_Perceptron()
    d_PA()