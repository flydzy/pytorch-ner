import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
sn.set(style="darkgrid")

from Preprocess import build_iterator

train_iter = build_iterator('data/en/eng.train', batch_size=32)
valid_iter = build_iterator('data/en/eng.testa', batch_size=32)
test_iter = build_iterator('data/en/eng.testb', batch_size=32)

def cal_length(data_iter, name):
    length_list = []
    tag_lists = []
    labels = []
    for batch in data_iter:
        length_list.extend([len(example.sent) for example in batch.dataset.examples])
        tag_lists.extend([example.label for example in batch.dataset.examples])
    
    for tag_list in tag_lists:
        if isinstance(tag_list, list):
            labels.extend(tag_list)
        else:
            labels.append(tag_list)
    
    labels_new = []
    for label in labels:
        if label == "O":
            continue
        else:
            labels_new.append(label)

    # print(length_list)
    sn.displot(length_list)
    
    plt.savefig(name + "_result.jpg")
    sn.countplot(y=labels_new)
    plt.savefig(name + "_tag.jpg")
    # plt.show()
    return  labels_new, length_list

labels1, length_list1 = cal_length(train_iter,"train")
labels2, length_list2 = cal_length(valid_iter,"valid")
labels3, length_list3 = cal_length(test_iter,"test")

length_list = []
label_list = []
length_list.extend(length_list1)
length_list.extend(length_list2)
length_list.extend(length_list3)
label_list.extend(labels1)
label_list.extend(labels2)
label_list.extend(labels3)
sn.displot(length_list)
plt.savefig("result_all.jpg")
sn.countplot(y=label_list)
plt.savefig("result_tag.jpg")