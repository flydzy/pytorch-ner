import sys
sys.path.append(r'..')
from Advtrans.preprocess import build_dataset
from Advtrans.models.crf import CRFModel

from sklearn.metrics import classification_report, precision_recall_fscore_support

en_dataset = build_dataset(train_path=r"data/en/eng.train", valid_path=r"data/en/eng.testa", test_path=r"data/en/eng.testb")

# Get Origin Sentence and Label
origin_dataset = {}
for key in en_dataset:
    train_sents = [example.sent for example in en_dataset[key].examples]
    train_labels = [example.label for example in en_dataset[key].examples]
    origin_dataset[key] = (train_sents, train_labels)

# load model
id2tags = en_dataset['train'].fields['label'].vocab.itos
tag2ids = en_dataset['train'].fields['label'].vocab.stoi  # N : num class for hmm
word2ids = en_dataset['train'].fields['sent'].vocab.stoi  # M : size of vocab for hmm

hmm = CRFModel()
hmm.train(origin_dataset['train'][0], origin_dataset['train'][1])
print("Training finished")

# validate
def accuracy(pred, gold):
    # flat prediction and gold label
    pred_flat = []
    gold_flat = []
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            pred_flat.append(pred[i][j])
            gold_flat.append(gold[i][j])
    return classification_report(gold_flat, pred_flat)


valid_pred_list = hmm.test(origin_dataset['valid'][0])
print("Valid Score:")
print(accuracy(valid_pred_list, origin_dataset['valid'][1]))


test_pred_list = hmm.test(origin_dataset['test'][0])
print("Test Score:")
print(accuracy(test_pred_list, origin_dataset['test'][1]))
