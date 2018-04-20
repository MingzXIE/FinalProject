
from numpy import *


def sigmoid(inx):
    return 1.0/(1+exp(-inx))

def grad_ascent(data_mat_in,class_labels):# 100,3 matrix
    data_matrix=mat(data_mat_in) # change to numpy matrix ,different features for col &sample for row
    label_mat=mat(class_labels).transpose()
    m,n=shape(data_matrix)
    # parameter for train
    alpha=0.001 # step length
    max_cycles=500 # iteration num
    weights=ones((n,1))
    for k in range(max_cycles):
        h=sigmoid(data_matrix * weights)
        error = (label_mat-h) # calculate the difference between real type and predict type
        weights = weights + alpha * data_matrix.transpose()*error
    return weights  # return the best parameter


def stoc_grad_ascent1(data_matrix, class_labels , num_iter=150):
    m,n = shape(data_matrix)
    weights = ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01 # alpha will descent as iteration rise, but will not be 0
            rand_index = int(random.uniform(0, len(data_index)))
            h=sigmoid(sum(data_matrix[rand_index]*weights))
            error=class_labels[rand_index]-h
            weights=weights + alpha*error * data_matrix[rand_index]
            del(data_index[rand_index])
    return weights


def classifyVector(in_x, weights):
    prob = sigmoid(sum(in_x * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def de_predict(file1, file2):
    fr_train=open(file1)
    fr_test=open(file2)
    training_set=[]
    training_labels=[]
    for line in fr_train.readlines():
        curr_line=line.strip().split('\t')
        line_Arr=[]
        for i in range(21):
            line_Arr.append(float(curr_line[i]))
        training_set.append(line_Arr)
        training_labels.append(line_Arr)
    train_weights=stoc_grad_ascent1(array(training_set),training_labels,500)
    error_count=0; num_test_vec=0.0
    for line in fr_test.readlines():
        num_test_vec+=1.0
        curr_line=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(curr_line[i]))
        if int(classifyVector(array(lineArr),train_weights))!=int(curr_line[21]):
            error_count+=1
    error_rate=(float(error_count)/num_test_vec)
    print("the error rate of this test is:%f" % error_rate)
    return error_rate


#
def multi_test(file1,file2):
    num_tests = 10;
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += de_predict(file1,file2)
    print
    "after %d iterations the average error rate is: %f" % (num_tests, error_sum / float(num_tests))

de_predict('logisticregtraining.txt', 'logisticregtesting.txt')
multi_test('logisticregtraining.txt', 'logisticregtesting.txt')