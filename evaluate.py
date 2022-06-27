from sklearn.metrics import classification_report, precision_recall_fscore_support
from preprocess import build_dataset
import torch


def align_predictions(predictions, golds, id2labels):
    """
    predictions: [Batch*Length Class]
    labels:[Batch*Length]
    """
    preds_list, labels_list = [], []
    for i in range(len(golds)):
        if golds[i] > 0:
            labels_list.append(id2labels[golds[i]])
            preds_list.append(id2labels[predictions[i]])
    return preds_list, labels_list

# hmm model evaluation 
def hmm_train_eval(origin_dataset, model, word2idx, tag2idx, logger):
    """
    train hmm and evaluate
    """
    # train
    model.train(origin_dataset['train'][0], origin_dataset['train'][1], word2idx, tag2idx)
    # evaluate
    train_result = model.test(origin_dataset['train'][0],word2id=word2idx, tag2id=tag2idx)
    dev_result = model.test(origin_dataset['dev'][0],word2id=word2idx, tag2id=tag2idx)
    test_result = model.test(origin_dataset['test'][0],word2id=word2idx, tag2id=tag2idx)
    # flatten the result
    train_result = [item for sublist in train_result for item in sublist]
    dev_result = [item for sublist in dev_result for item in sublist]
    test_result = [item for sublist in test_result for item in sublist]
    golds = origin_dataset['test'][1]
    # flatten the glods
    golds = [item for sublist in golds for item in sublist]
    logger.info("\n" + classification_report(golds, test_result))

def crf_train_eval(origin_dataset, model, logger):
    """
    train crf and evaluate
    """
    # train
    model.train(origin_dataset['train'][0], origin_dataset['train'][1])
    # evaluate
    train_result = model.test(origin_dataset['train'][0])
    dev_result = model.test(origin_dataset['dev'][0])
    test_result = model.test(origin_dataset['test'][0])
    # flatten the result
    train_result = [item for sublist in train_result for item in sublist]
    dev_result = [item for sublist in dev_result for item in sublist]
    test_result = [item for sublist in test_result for item in sublist]
    golds = origin_dataset['test'][1]
    # flatten the glods
    golds = [item for sublist in golds for item in sublist]
    logger.info("\n" + classification_report(golds, test_result))

def eval_of_bilstm(model, dataiter, id2labels, out_size):
    all_preds, all_labels = [], []
    for batch in dataiter:
        labels = batch.label
        labels = labels.view(-1)
        output = model(batch.sent[0], batch.sent[1]).view(-1, out_size)
        golds = labels.cpu().numpy()
        predictions = torch.argmax(output, dim=1).cpu().numpy()
        preds_list, labels_list = align_predictions(predictions, golds, id2labels)
        all_preds.extend(preds_list)
        all_labels.extend(labels_list)
    return all_preds, all_labels


def lstm_train_eval(dataiters, model, loss_function, optimizer, epoches, out_size, id2labels, logger):
    """
    train lstm and evaluate
    """
    # train
    EPOCHES = epoches
    for epoch in range(EPOCHES):
        for step, batch in enumerate(dataiters['train']):
            sents = batch.sent
            labels = batch.label.view(-1)

            output = model(sents[0], sents[1]).view(-1, out_size)
            loss = loss_function(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                logger.info("Epoch [{}/{}], Step [{}/{}], loss:{:.4f}".format(epoch+1, EPOCHES, step+1, len(dataiters['train']),loss.item()))
        
        # evaluate each epoch
        with torch.no_grad():
            train_preds,train_labels = eval_of_bilstm(model, dataiters['train'], id2labels, out_size)
            valid_preds,valid_labels = eval_of_bilstm(model, dataiters['dev'], id2labels, out_size)
            test_preds,test_labels = eval_of_bilstm(model, dataiters['test'], id2labels, out_size)
            # output the result
            logger.info(precision_recall_fscore_support(train_labels, train_preds, average='macro'))
            logger.info(precision_recall_fscore_support(valid_labels, valid_preds, average='macro'))
            logger.info(precision_recall_fscore_support(test_labels, test_preds, average='macro'))

    # evaluate
    with torch.no_grad():
        final_pres, final_labels = eval_of_bilstm(model, dataiters['test'], id2labels, out_size)
        logger.info("\n" + classification_report(final_labels, final_pres))

def get_mask(text, text_len):
    mask = torch.zeros(text.shape, dtype=torch.bool)
    for i in range(len(text_len)):
        mask[i, :text_len[i]] = 1
    return mask

def eval_of_bilstm_crf(model, dataiter, id2labels):
    all_preds, all_labels = [], []
    for batch in dataiter:
        text = batch.sent
        label = batch.label
        mask = get_mask(text[0], text[1])
        preds = model(text[0], text[1], tags = None, mask = mask)
        preds = [label for pred in preds for label in pred ]
        golds = label.masked_select(mask).cpu().numpy()
        preds_list, labels_list = align_predictions(preds, golds, id2labels)
        all_preds.extend(preds_list)
        all_labels.extend(labels_list)
    return all_labels, all_preds


def lstm_crf_train_eval(dataiters, model, optimizer, epoches, id2labels ,logger):
    """
    train lstm and evaluate
    """
    # train
    EPOCHES = epoches
    for epoch in range(EPOCHES):
        for step, batch in enumerate(dataiters['train']):
            text = batch.sent
            label = batch.label
            # text[1]表是batch中的每一个句子的原始长度
            # 根据原始长度生成mask
            mask = torch.zeros(text[0].shape, dtype=torch.bool)
            for i in range(len(text[1])):
                mask[i, :text[1][i]] = 1
            loss = model(text[0], text[1], tags = label, mask = mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                logger.info("Epoch [{}/{}], Step [{}/{}], loss:{:.4f}".format(epoch+1, EPOCHES, step+1, len(dataiters['train']),loss.item()))
        
        # evaluate each epoch
        train_preds,train_labels = eval_of_bilstm_crf(model, dataiters['train'], id2labels)
        valid_preds,valid_labels = eval_of_bilstm_crf(model, dataiters['dev'],id2labels)
        test_preds,test_labels = eval_of_bilstm_crf(model, dataiters['test'],id2labels)
        # output the result
        logger.info(precision_recall_fscore_support(train_labels, train_preds, average='macro'))
        logger.info(precision_recall_fscore_support(valid_labels, valid_preds, average='macro'))
        logger.info(precision_recall_fscore_support(test_labels, test_preds, average='macro'))

    # evaluate
    with torch.no_grad():
        final_pres, final_labels = eval_of_bilstm_crf(model, dataiters['test'], id2labels)
        logger.info("\n" + classification_report(final_labels, final_pres))


if __name__ == '__main__':
    pass
