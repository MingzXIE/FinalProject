from numpy import *

# def load_dataset(file_name):
#     data_matrix=[];label_matrix=[]
#     fr=open(file_name)
#     for line in fr.readlines():
#         line_array = line.strip().split()
#         data_matrix.append([1.0,float(line_array[0]),float(line_array[1])])
#         label_matrix.append(int(line_array[2]))
#     return data_matrix,label_matrix


def sigmoid(inx):
    return 1.0/(1+exp(-inx))


def grad_ascent(data_matrix_input, labels_input, iterate_num):
    # convert the set into numpy matrix
    data_matrix=mat(data_matrix_input)
    label_mat = mat(labels_input).transpose()
    m, n = shape(data_matrix)
    alpha = 0.001
    max_cycles=500#iteration num
    weights=ones((n,1))
    # iterate
    for k in range(max_cycles):
        h=sigmoid(data_matrix*weights) #h is a vector
        error=(label_mat-h) #compute the difference between real type and predict type
        weights=weights+alpha*data_matrix.transpose()*error
    return weights #return the best parameter


def stoc_grad_ascent(data_matrix,class_labels,num_iter=150):
    m,n=shape(data_matrix)
    weights=ones(n)
    for j in range(num_iter):
        data_index = list(range(m))# python3 change: dataIndex=range(m)
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01 #alpha will descent as iteration rise,but does not be 0
            rand_index = int(random.uniform(0, len(data_index)))
            h=sigmoid(sum(data_matrix[rand_index]*weights))
            error=class_labels[rand_index]-h
            weights=weights+alpha*error*data_matrix[rand_index]
            del(data_index[rand_index])
    return weights


def classify_vector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def logistic_reg_train(train_file, test_file):
    fr_train = open(train_file)
    fr_test = open(test_file)
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        split_line = line.strip().split('\t')
        line_arr =[]
        for i in range(21):
            line_arr.append(float(split_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(split_line[21]))
    train_weights = stoc_grad_ascent(array(training_set), training_labels, 500)
    error_count = 0; num_test_vec = 0.0
    for line in fr_test.readlines():
        num_test_vec += 1.0
        currLine = line.strip().split('\t')
        line_arr =[]
        for i in range(21):
            line_arr.append(float(currLine[i]))
        if int(classify_vector(array(line_arr), train_weights))!= int(currLine[21]):
            error_count += 1
    error_rate = (float(error_count)/num_test_vec)
    print("the error rate of this test is: %f" % error_rate)
    return error_rate


def multi_test(train_file, test_file):
    num_tests = 10; error_sum=0.0
    for k in range(num_tests):
        error_sum += logistic_reg_train(train_file, test_file)
    print("after %d iterations the average error rate is: %f" % (num_tests, error_sum/float(num_tests)))


logistic_reg_train('logisticregtraining.txt', 'logisticregtesting.txt')
multi_test('logisticregtraining.txt', 'logisticregtesting.txt')