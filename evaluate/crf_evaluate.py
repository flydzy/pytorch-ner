import sys
sys.path.append(r'..')
from Advtrans.preprocess.preprocess import build_iterator
from Advtrans.models.crf import CRFModel, word2features, sent2features

train_iter = build_iterator('data/en/eng.train', batch_size=32)
valid_iter = build_iterator('data/en/eng.testa', batch_size=32)
test_iter = build_iterator('data/en/eng.testb', batch_size=32)


def getOriginData(data_iter, part=1):
    sents = []
    labels = []
    for batch in data_iter:
        sents.extend([example.sent for example in batch.dataset.examples])
        labels.extend([example.label for example in batch.dataset.examples])
    return sents[:int(len(sents)*part)], labels[:int(len(labels)*part)]

# 获取原始数据并展开
dataset = {}
dataset['train'] = getOriginData(train_iter,part=0.01)
dataset['valid'] = getOriginData(valid_iter,part=0.01)
dataset['test'] = getOriginData(test_iter,part=0.01)

# 运用CRF模型训练
print("Training CRF...")
model = CRFModel()
model.train(dataset['train'][0], dataset['train'][1])
# 评估模型
def accuracy(pred_list, tag_list):
    correct = 0
    allcount = 0
    for i in range(len(pred_list)):
        for j in range(len(pred_list[i])):
            if pred_list[i][j] == tag_list[i][j]:
                correct += 1
        allcount += len(pred_list[i])
    return correct / allcount

print("Training finished")
pred_list = model.test(dataset['valid'][0])
print("Eval Accuracy: {}".format(accuracy(pred_list, dataset['valid'][1])))

# 测试集测试
pred_list = model.test(dataset['test'][0])
print("Test Accuracy: {}".format(accuracy(pred_list, dataset['test'][1])))

