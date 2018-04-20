from numpy import *
import time

def load_dataset(file_name, k):
    data_matrix = []
    label_matrix = []
    fr = open(file_name)
    for line in fr.readlines():
        line_array = line.strip().split('\t')
        if k == 2:
            data_matrix.append([float(line_array[0]), float(line_array[1])])
            label_matrix.append(float(line_array[2]))
        elif k == 3:
            data_matrix.append([float(line_array[0]), float(line_array[1])], float(line_array[2]))
            label_matrix.append(float(line_array[-1]))
        elif k == 4:
            data_matrix.append([float(line_array[0]), float(line_array[1])], float(line_array[2]), float(line_array[3]))
            label_matrix.append(float(line_array[-1]))
        elif k == 5:
            data_matrix.append([float(line_array[0]), float(line_array[1])], float(line_array[2]), float(line_array[3]), float(line_array[4]))
            label_matrix.append(float(line_array[-1]))

    return data_matrix,label_matrix


def select_rand(i,m):
    j = i
    while j == i:
        j = int(random.uniform(0,m))
    return j


def clip_alpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smo(data_matrix_in, class_labels, constant, tolerate, iterate):
    data_matrix = mat(data_matrix_in); label_matrix = mat(class_labels).transpose()
    b = 0; m,n = shape(data_matrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < iterate):
        alpha_pairs_changed = 0
        for i in range(m):
            fXi = float(multiply(alphas,label_matrix).T*(data_matrix * data_matrix[i,:].T)) + b
            Ei = fXi - float(label_matrix[i])
            if ((label_matrix[i]*Ei < -tolerate) and (alphas[i] < constant)) or ((label_matrix[i]*Ei > tolerate) and (alphas[i] > 0)):
                j = select_rand(i,m)
                fXj = float(multiply(alphas,label_matrix).T*(data_matrix * data_matrix[j,:].T)) + b
                Ej = fXj - float(label_matrix[j])
                alpha_Iold = alphas[i].copy()
                alpha_Jold = alphas[j].copy();
                if (label_matrix[i] != label_matrix[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(constant, constant + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - constant)
                    H = min(constant, alphas[j] + alphas[i])
                if L==H:
                    print("L==H")
                    continue

                eta = 2.0 * data_matrix[i,:] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i,:].T - data_matrix[j,:] * data_matrix[j,:].T
                if eta >= 0: print("eta>=0"); continue
                alphas[j] -= label_matrix[j]*(Ei - Ej)/eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - alpha_Jold) < 0.00001:
                    print("j not moving enough")
                    continue
                alphas[i] += label_matrix[j] * label_matrix[i] * (alpha_Jold - alphas[j])
                b1 = b - Ei- label_matrix[i]*(alphas[i]-alpha_Iold) * data_matrix[i,:] * data_matrix[i,:].T - label_matrix[j] * (alphas[j]-alpha_Jold) * data_matrix[i,:] * data_matrix[j,:].T
                b2 = b - Ej- label_matrix[i]*(alphas[i]-alpha_Iold) * data_matrix[i,:] * data_matrix[j,:].T - label_matrix[j]*(alphas[j]-alpha_Jold)*data_matrix[j,:]*data_matrix[j,:].T
                if (0 < alphas[i]) and (constant > alphas[i]): b = b1
                elif (0 < alphas[j]) and (constant > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alpha_pairs_changed += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i , alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas

start_time = time.time()

data, label = load_dataset('svm_testSet.txt', 2)
b, alphas = smo(data, label, 0.6, 0.001, 40)
print(b)
shape(alphas[alphas>0])

end_time = time.time()
print(end_time - start_time)