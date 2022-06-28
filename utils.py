def align_predictions(predictions, golds, id2labels):
    """
    predictions: [Batch*Length Class]
    labels:[Batch*Length]
    """
    preds_list, labels_list = [], []
    for i in range(len(golds)):
        if golds[i] != -100:
            labels_list.append(id2labels[golds[i]])
            preds_list.append(id2labels[predictions[i]])
    return preds_list, labels_list