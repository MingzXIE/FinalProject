from numpy import *


# Create the vocabulary list
def create_vocabulary_list(dataset):
    vocabulary_set = set([])
    for doc in dataset:
        vocabulary_set = vocabulary_set | set(doc)
    return list(vocabulary_set)


# convert words to vector
def words_to_vector(vocabulary_list, input_set):
    vector_created = [0] * len(vocabulary_list)
    for word in input_set:
        if word in vocabulary_list:
            vector_created[vocabulary_list.index(word)] = 1
        else:
            print('the word: %s is not in my vocabulary! ' % 'word')
    return vector_created


def train_nb(matrix_to_train, labels):
    number_of_docs_to_train = len(matrix_to_train)
    number_of_words = len(matrix_to_train[0])
    abusive_probability = sum(labels) / float(number_of_docs_to_train)
    frequency_0 = ones(number_of_words)
    frequency_1 = ones(number_of_words)
    probability_0_denominator = 2.0
    probability_1_denominator = 2.0
    for i in range(number_of_docs_to_train):
        if labels[i] == 1:
            frequency_1 += matrix_to_train[i]
            probability_1_denominator += sum(matrix_to_train[i])
        else:
            frequency_0 += matrix_to_train[i]
            probability_0_denominator += sum(matrix_to_train[i])
    probability_1_vector = log(frequency_1 / probability_1_denominator)
    probability_0_vector = log(frequency_0 / probability_0_denominator)
    return probability_0_vector, probability_1_vector, abusive_probability


def nb_classify(vector_to_classify, frequency_0_vector, frequency_1_vector, probability_class_1):
    probability_1 = sum(vector_to_classify * frequency_1_vector) + log(probability_class_1)
    probability_0 = sum(vector_to_classify * frequency_0_vector) + log(1.0 - probability_class_1)
    if probability_0 > probability_1:
        return 0
    else:
        return 1


def bag_of_words_model(vocabulary_list, input_set):
    vector_created = [0] * len(vocabulary_list)
    for word in input_set:
        if word in vocabulary_list:
            vector_created[vocabulary_list.index(word)] += 1
        else:
            print('the word: %s is not in my vocabulary! ' % 'word')
    return vector_created


# create the word list
def text_parse(big_string):
    import re
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_test(number_of_files, spam_folder_name, ham_folder_name, testing_rate):
    doc_list = []
    label_list = []
    full_text = []
    for i in range(1,26,1):
        word_list = text_parse(open('/Users/Major/Documents/AI/machinelearninginaction/Ch04/email/spam/%d.txt' %i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        # label the spam 1
        label_list.append(1)
        word_list = text_parse(open('/Users/Major/Documents/AI/machinelearninginaction/Ch04/email/spam/%d.txt' %i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        # label the ham 0
        label_list.append(0)
    vocabulary_list = create_vocabulary_list(doc_list)  # create vocabulary list

    # select testing set randomly, others for training
    training_set = range(50)
    test_set = []
    number_of_testing = number_of_files * testing_rate
    for i in range(number_of_testing):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])

    # create training set
    train_matrix = []
    train_labels = []
    for doc_index in training_set:
        train_matrix.append(bag_of_words_model(vocabulary_list, doc_list[doc_index]))
        train_labels.append(label_list[doc_index])
    p0V, p1V, pSpam = train_nb(array(train_matrix), array(train_labels))
    # test and calculate error rate
    error_count = 0
    for doc_index in test_set:
        wordVector = bag_of_words_model(vocabulary_list, doc_list[doc_index])
        if nb_classify(array(train_matrix), p0V, p1V, pSpam) != label_list[doc_index]:
            error_count += 1
            print("classification error", doc_list[doc_index])
    print('the error rate is: ', float(error_count) / len(test_set))


spam_test(25, 'nbtesting/spam/', 'nbtesting/ham/', 0.1)




