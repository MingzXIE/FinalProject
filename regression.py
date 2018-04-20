from numpy import *

def load_dataset(file_name):
    num_of_feature = len(open(file_name).readline().split('\t')) - 1
    data_matrix = []; label_matrix = []
    fr = open(file_name)
    for line in fr.readlines():
        line_array =[]
        cur_line = line.strip().split('\t')
        for i in range(num_of_feature):
            line_array.append(float(cur_line[i]))
        data_matrix.append(line_array)
        label_matrix.append(float(cur_line[-1]))
    return data_matrix,label_matrix


def stand_regression(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


x_array, y_array = load_dataset('regressiontesting.txt')
print(x_array)
print(y_array)
ws = stand_regression(x_array, y_array)
print(ws)

user_input = [1.0, 0.92577]
x_input = mat(user_input)
y_predict = x_input * ws
print(y_predict)

