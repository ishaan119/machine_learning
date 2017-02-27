'''
Created on May 5, 2016

@author: ishaansutaria
'''
import minst as ag
import numpy as np
from matplotlib import pyplot as plt
import math
import class_2d_hist

path_training = '/Users/rajeshsutaria/git/machine_learning/PCA_DIGIT_RECOGNITION/minst_db/'


def display_image(matrix, show=True):
    if show:
        plt.imshow(matrix)
        plt.show()


def plot_vector(array):
    fg, ax = plt.subplots()
    plt.plot(range(len(array)), array, linestyle='-', c="b")
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_title("Mean image plot")
    plt.show()


def cal_PCA(digit_array):
    oned_images = []
    sample_len = {}
    for d in digit_array:
        images, labels = ag.load_mnist('training', digits=[
                                       d], path=path_training)
        sample_len[d] = len(images)
        print 'Display an Image from the traning set'
        display_image(images[0])
        print 'Compute the data set Matrix'
        for i in images:
            oned_images.append(i.flatten())
    X = np.array(oned_images)
    print 'Dimension of X Matrix is {}'.format(X.shape)
    mu_x = np.mean(X, axis=0)
    plot_vector(mu_x)
    print 'Dimension of mean vector {}'.format(mu_x.shape)
    print 'Min of mean vector is {0} and max is {1}'.format(min(mu_x), max(mu_x))
    display_image(mu_x.reshape(28, 28))
    Z = X - mu_x
    print 'Dimension of Z matrix is {}'.format(Z.shape)
    Z_mean = Z.mean(axis=0)
    print 'Is mean vector of Z all zeroes: {}'.format(np.all(Z_mean == 0))
    print 'Min of Z should be negative actual value is {}'.format(np.amin(Z))
    print 'Max of Z should be positive actual value is {}'.format(np.amax(Z))
    C = np.ma.cov(Z, rowvar=False)
    C = np.array(C)
    print 'Dimensions of the co_variance_matrix is {}'.format(C.shape)
    display_image(C)
    eg_values, V = np.linalg.eig(C)
    V = V.real
    # Testing if Eigen vector is normalized.
    print "Eigen normalized value-1:", np.linalg.norm(V[[50]])
    print 'Dimensions of V are {}'.format(V.shape)
    V = np.matrix(V).transpose()
    TV = np.matrix(V).transpose()
    P = Z * TV
    P_MEAN = P.mean(axis=0)
    P1P2 = P[:, 0:2]
    V1V2 = V[0:2, :]
    R = P1P2 * V1V2
    P1P2 = np.array(P1P2)
    no_r, no_c = np.shape(P1P2)
    x_axis = P1P2[:, 0]
    y_axis = P1P2[:, 1]
    NR = R + mu_x
    remaining_cnt = no_r - sample_len[7]
    '''
    display_image(NR[0].reshape(28, 28), True)
    f, ax = plt.subplots()
    for i in range(sample_len[7]):
        plt_1 = ax.scatter(x_axis[i], y_axis[i], c="r", marker='o', label="Seven")
    for i in range(remaining_cnt):
        plt_2 = ax.scatter(x_axis[i+sample_len[7]], y_axis[i+sample_len[7]], c="b", marker='o', label="Six")
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_title("Digits - Seven & Six")
    plt.legend([plt_1, plt_2], ["Seven", "Six"])
    plt.show()
    # Why add the mean vector back?
    NR = R + mu_x
    display_image(NR[0].reshape(28, 28), True)
    '''
    return P1P2, sample_len[7], remaining_cnt, mu_x, V1V2, sample_len


def compute_bayseian_classifier(n, x, y, mean_vector, co_variance_matrix):
    determinant_sigma_co_variance = np.linalg.det(co_variance_matrix)
    sqrt_determinant_sigma_co_variance = math.sqrt(
        determinant_sigma_co_variance)
    n_by_sqrt_determinant_sigma_co_variance = n / sqrt_determinant_sigma_co_variance
    n_dim_sample_minus_mean_vector = ((x, y) - mean_vector)
    inverse_sigma_co_variance = np.linalg.inv(co_variance_matrix)
    transpose_n_dim_sample_minus_mean_vector = np.matrix(
        n_dim_sample_minus_mean_vector).transpose()
    return_value = (n_by_sqrt_determinant_sigma_co_variance * np.exp((-1 / 2) * np.matrix(
        n_dim_sample_minus_mean_vector) * inverse_sigma_co_variance * transpose_n_dim_sample_minus_mean_vector))
    return return_value


def find_regular_probability(d1_count, d2_count):
    d1_prob = 0
    d2_prob = 0
    t_count = d1_count + d2_count
    if (t_count == 0):
        # print "The probability can't be determined with given training set"
        return 3

    if (d1_count != 0):
        d1_prob = d1_count / t_count

    if (d2_count != 0):
        d2_prob = d2_count / t_count

    # print "D1-count:%f, D2-count:%f, Total-count:%d"
    # %(d1_count,d2_count,t_count)
    if (d1_prob > d2_prob):
        # print "The probability is Digit-1"
        return 1
    elif (d2_prob > d1_prob):
        # print "The probability is Digit-2"
        return 2
    else:
        # print "The probability can't be determined with given training set"
        return 3
    return


arr1 = [7, 6]
R_PCA, N1, N2, mean, eigen_vector, sample_len = cal_PCA(arr1)
eigen_vector = np.array(eigen_vector)
no_r, no_c = np.shape(R_PCA)
x_axis = R_PCA[:, 0]
y_axis = R_PCA[:, 1]
f, ax = plt.subplots()
splice_start = 0
t_list = []
for i in arr1:
    t_list.append((splice_start + sample_len[i]))
    splice_start += sample_len[i]

# for i in
print "***********TRAINING DONE*****************"

# Precomputing for baysian
first_digit_pca = R_PCA[0:N1, :]
second_digit_pca = R_PCA[(N1 + 1):(N1 + N2), :]

d1_mean = np.mean(first_digit_pca, axis=0)
d1_cov = np.cov(first_digit_pca, rowvar=False)
d2_mean = np.mean(second_digit_pca, axis=0)
d2_cov = np.cov(second_digit_pca, rowvar=False)

# Precomputing for histogram
x_axis = R_PCA[:, 0]
y_axis = R_PCA[:, 1]
min_x = min(x_axis)
max_x = max(x_axis)
min_y = min(y_axis)
max_y = max(y_axis)
bin_size = 15

'''
d1_inst = class_2d_hist.histogram_2d(
    first_digit_pca, min_x, max_x, min_y, max_y, bin_size)
d2_inst = class_2d_hist.histogram_2d(
    second_digit_pca, min_x, max_x, min_y, max_y, bin_size)
d1_2d_hist = d1_inst.get_2d_histogram()
d2_2d_hist = d2_inst.get_2d_histogram()
'''

images, labels = ag.load_mnist(
    'testing', digits=arr1, path=path_training)
n_samples_tested = 0
correctly_tested = 0
incorrect_prediction = 0
baysian_result_dic = {'correct_7': 0,
                      'incorrect_7': 0, 'correct_6': 0, 'incorrect_6': 0}
histogram_result_dic = {'correct_7': 0,
                        'incorrect_7': 0, 'correct_6': 0, 'incorrect_6': 0}
b_t_6 = 0
b_t_7 = 0
b_h_6 = 0
b_h_7 = 0
for img, label in zip(images, labels):
    x = img.flatten()
    z = x - mean
    P1P2 = z * (np.matrix(eigen_vector).transpose())
    P1P2 = np.array(P1P2)
    p7 = compute_bayseian_classifier(
        N1, P1P2[0][0], P1P2[0][1], d1_mean, d1_cov)
    p6 = compute_bayseian_classifier(
        N2, P1P2[0][0], P1P2[0][1], d2_mean, d2_cov)
    n_samples_tested += 1
    if p7 > p6:
        result = 7
    else:
        result = 6
    if result == label:
        if b_t_6 == 0 and label == 6:
            print 'Correct Bayes for Digit 6'
            display_image(img, True)
            b_t_6 = 1
        if b_t_7 == 0 and label == 7:
            print 'Correct Bayes for Digit 7'
            display_image(img, True)
            b_t_6 = 1
        baysian_result_dic['correct_' + str(label)] += 1
    else:
        if b_t_6 == 1 and label == 6:
            print 'InCorrect Bayes for Digit 6'
            display_image(img, True)
            b_t_6 = 2
        if b_t_7 == 1 and label == 7:
            print 'InCorrect Bayes for Digit 6'
            display_image(img, True)
            b_t_7 = 2
        baysian_result_dic['incorrect_' + str(label)] += 1
'''
    d1_count = d1_inst.get_bin_count(P1P2[0][0], P1P2[0][1])
    d2_count = d2_inst.get_bin_count(P1P2[0][0], P1P2[0][1])
    prob_rc = find_regular_probability(d1_count, d2_count)
    if prob_rc == 1:
        result = 7
    else:
        result = 6
    if result == label:
        if b_h_6 == 0 and label == 6:
            print 'Correct Histogram for Digit 6'
            display_image(img, True)
            b_h_6 = 1
        if b_h_7 == 0 and label == 7:
            print 'Correct Histogram for Digit 7'
            display_image(img, True)
            b_h_6 = 1
        histogram_result_dic['correct_' + str(label)] += 1
    else:
        if b_h_6 == 1 and label == 6:
            print 'InCorrect Histogram for Digit 6'
            display_image(img, True)
            b_h_6 = 2
        if b_h_7 == 1 and label == 7:
            print 'InCorrect Histogram for Digit 6'
            display_image(img, True)
            b_h_7 = 2
        histogram_result_dic['incorrect_' + str(label)] += 1
'''

print 'Total Number of samples tested: {}'.format(n_samples_tested)
print 'Reulsts using Baysian Classifier'
print baysian_result_dic
# print 'Results from Histogram'
# print histogram_result_dic
