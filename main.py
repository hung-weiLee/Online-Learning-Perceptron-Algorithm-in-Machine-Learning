#!/usr/bin/python3
import Binary_Perceptron
import Binary_PA
import Binary_averaged_perceptron
import Binary_General_learning_curve
################################################
import Multi_Perceptron
import Multi_PA
import Multi_averaged_perceptron
import Multi_General_learning_curve


def main():
    print("***(Binary Classification)***")

    print("======= (problem a.) =======")  # a
    Binary_Perceptron.a()
    Binary_PA.a()
    print("======= (finish  a.) =======")

    print("======= (problem b.) =======")  # b
    Binary_Perceptron.b()
    Binary_PA.b()
    print("======= (finish  b.) =======")

    print("======= (problem c.) =======")  # c
    Binary_averaged_perceptron.c()
    print("plain perceptron test accuracies :", Binary_Perceptron.accuracy_test.pop())
    print("======= (finish  c.) =======")

    print("======= (problem d.) =======")  # d
    Binary_General_learning_curve.d_Perceptron()
    Binary_General_learning_curve.d_PA()
    print("======= (finish  d.) =======")

    print('\n')  ################################################

    print("***Multi-Class Classification***")

    print("======= (problem a.) =======")  # a
    Multi_Perceptron.a()
    Multi_PA.a()
    print("======= (finish  a.) =======")

    print("======= (problem b.) =======")  # b
    Multi_Perceptron.b()
    Multi_PA.b()
    print("======= (finish  b.) =======")

    print("======= (problem c.) =======")  # c
    Multi_averaged_perceptron.c()
    print("plain perceptron test accuracies :", Multi_Perceptron.accuracy_test.pop())
    print("======= (finish  c.) =======")

    print("======= (problem d.) =======")  # d
    Multi_General_learning_curve.d_Perceptron()
    Multi_General_learning_curve.d_PA()
    print("======= (finish  d.) =======")


if __name__ == '__main__':
    main()

