from numpy import *
import operator


def file_to_matrix(file_name, n): # n is the number of the attributes
    fr = open(file_name)

    # Get the number of lines
    lines = fr.readlines()
    number_of_lines = len(lines)

    # return the matrix
    matrix_created = zeros((number_of_lines, n))

    # return the labels
    classify_label_vector = []
    index = 0
    for line in lines:
        line = line.strip() # delete all the '\n'
        elements_list = line.split('\t')
        matrix_created[index, :] = elements_list[0: n]
        classify_label_vector.append(int(elements_list[-1])) # the last element is the label
        index += 1

    return matrix_created, classify_label_vector


# Principle of normalization: new_value = (old_value-min)/(max-min)
def normalization(dataset):
    min_value = dataset.min(0)
    max_value = dataset.max(0)
    ranges = max_value - min_value
    normalized_dataset = zeros(shape(dataset))
    m = dataset.shape[0]
    normalized_dataset = dataset - tile(min_value, (m,1))
    normalized_dataset = normalized_dataset/tile(ranges, (m,1))
    return normalized_dataset, ranges, min_value # return 3 values, one for training, the other two for testing


def knn_classify(unlabelled_data, dataset, labels, k):
    dataset_size = dataset.shape[0]

    # calculate distances and sort them
    diff_mat = tile(unlabelled_data, (dataset_size,1)) - dataset
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis = 1)
    distances = sq_distances ** 0.5
    sorted_distances = distances.argsort()
    class_count = {}

    # vote for the result
    for i in range(k):
        selected_label = labels[sorted_distances[i]]
        class_count[selected_label] = class_count.get(selected_label, 0)+1
    sorted_class_count = sorted(class_count.items(), key = operator.itemgetter(1), reverse = True)

    # classify the unlabelled data.

    return sorted_class_count[0][0]


# return the error rate
def knn_test(file_input, k, n, test_ratio):
    input_matrix, labels = file_to_matrix(file_input, n)
    normalized_matrix, ranges, minimum_value = normalization(input_matrix)
    m = normalized_matrix.shape[0]
    number_of_test = int(m * test_ratio)
    error_count = 0.0
    for i in range(number_of_test):
        classify_result = knn_classify(normalized_matrix[i, :], normalized_matrix[number_of_test:m, :], labels[number_of_test:m], k)
        print("the result classified by classifier is: %d, the real answer is: %d" % (classify_result, labels[i]))
        if classify_result != labels[i]:
            error_count += 1.0
    error_rate = error_count/float(number_of_test)
    return error_rate

real_error_rate = knn_test('testset/knntesting.txt', 3, 3, 0.5)
print(real_error_rate)


def string_to_list(string_input, n):
    line = string_input.strip()
    float_list=[]
    elements_list = line.split(',')
    for str_number in elements_list:
        float_num = float(str_number)
        float_list.append(float_num)

    return float_list



def knn_predict(array_to_predict, file_input, k, n):
    input_matrix, labels = file_to_matrix(file_input, n)
    normalized_matrix, ranges, minimum_value = normalization(input_matrix)
    predict_result = knn_classify((array_to_predict - minimum_value)/ranges, normalized_matrix, labels, k)
    return predict_result

# array_to_predict_input = [10.0, 10000.0, 0.5 ]
# result = knn_predict(array_to_predict_input, 'testset/knntesting.txt', 3,3)
# print(result)

# string_input = "10,10000.0,0.5 "
# ntext = 3
# matrix_test = string_to_list(string_input, ntext)
# result = knn_predict(matrix_test, 'testset/knntesting.txt', 3,3)
# print(result)
#
# print(matrix_test)











