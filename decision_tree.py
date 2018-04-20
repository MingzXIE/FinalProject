from math import log
import operator


def calculate_shannon_entropy(dataset):
    number_of_entries = len(dataset)
    label_counts = {}

    for feature_vector in dataset:
        current_label = feature_vector[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_entropy = 0.0

    for key in label_counts:
        probability = float(label_counts[key]) / number_of_entries
        shannon_entropy -= probability * log(probability, 2)
    return shannon_entropy


def split_dataset(dataset, axis, value):
    new_list = []
    for feature_vector in dataset:
        if feature_vector[axis] == value:
            reduced_feature = feature_vector[:axis]
            reduced_feature.extend(feature_vector[axis + 1:])
            new_list.append(reduced_feature)
    return new_list


def choose_best_feature_to_split(dataset):
    number_of_features = len(dataset[0]) - 1
    base_entropy = calculate_shannon_entropy(dataset)
    best_information_gained = 0.0
    best_feature_to_split = -1
    for i in range(number_of_features):
        feature_list = [example[i] for example in dataset]
        unique_vals = set(feature_list)
        new_entropy = 0.0    # calculate the entropy
        for value in unique_vals:
            sub_dataset = split_dataset(dataset, i, value)
            probability = len(sub_dataset)/float(len(dataset))
            new_entropy += probability * calculate_shannon_entropy(sub_dataset)
        information_gained = base_entropy - new_entropy
        if information_gained > best_information_gained:  # the more the better
            best_information_gained = information_gained
            best_feature_to_split = i
    return best_feature_to_split



# vote function
def vote_function (class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1

    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# Create the decision tree via recursion
def create_tree(dataset, labels):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0] # Stop splitting when all of the classes are equal

    if len(dataset[0]) == 1: # Stop splitting when there are no more features in the dataset
        return vote_function(class_list)

    best_feature = choose_best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    decision_tree = {best_feature_label : {}}
    sub_labels = labels[:]

    del (sub_labels[best_feature])
    feat_value = [example[best_feature] for example in dataset]
    unique_values = set(feat_value)
    for value in unique_values:
        decision_tree[best_feature_label][value] = create_tree(split_dataset(dataset, best_feature, value), sub_labels)

    return decision_tree


def dt_predict(trained_tree, label_list, vector_to_predict):
    first_side = list(trained_tree.keys())
    first_str = first_side[0]
    second_dict = trained_tree[first_str]

    # Change strings to index
    feature_index = label_list.index(first_str)
    key = vector_to_predict[feature_index]
    feature_value = second_dict[key]

    if isinstance(feature_value, dict):
        class_label = dt_predict(feature_value, label_list, vector_to_predict)
    else:
        class_label = feature_value
    return class_label


# store and grab the tree with file
def store_tree(input_tree, file_name):
    import pickle
    fw = open(file_name, 'wb+')
    pickle.dump(input_tree, fw)
    fw.close()


def grab_tree(file_name):
    import pickle
    fr = open(file_name, 'rb')
    return pickle.load(fr)


def file_to_tree(file_name, label_list):
    fr = open(file_name)
    target_dataset = [inst.strip().split('\t') for inst in fr.readlines()]
    target_tree = create_tree(target_dataset, label_list)
    return target_tree




new_label = ['age', 'prescript', 'astigmatic', 'tearRate']
new_tree = file_to_tree('decisiontreetesting.txt', new_label)
print(dt_predict(new_tree, new_label, ['presbyopic', 'hyper', 'yes', 'normal']))
