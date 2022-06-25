import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from preprocess import build_dataset
sns.set(style="darkgrid")

en_dataset = build_dataset(train_path=r"data/en/eng.train", valid_path=r"data/en/eng.testa", test_path=r"data/en/eng.testb")

# 统计各个数据集中的标签数
train_labels = [example.label for example in en_dataset['train'].examples]
valid_labels = [example.label for example in en_dataset['valid'].examples]
test_labels = [example.label for example in en_dataset['test'].examples]

# 展开
train_labels = [label for label_list in train_labels for label in label_list if label != 'O']
valid_labels = [label for label_list in valid_labels for label in label_list if label != 'O']
test_labels = [label for label_list in test_labels for label in label_list if label != 'O']

# 绘图
fix, ax = plt.subplots(1,3,figsize=(20, 4))
sub1 = sns.countplot(train_labels, ax=ax[0])
sub1.set_title('Train')
sub2 = sns.countplot(valid_labels, ax=ax[1])
sub2.set_title('Valid')
sub3 = sns.countplot(test_labels, ax=ax[2])
sub3.set_title('Test')
plt.savefig('imgs/statistic_tags_train.png')
