
# hmm model evaluation 
def hmm__train_eval(train_iter, dev_iter, test_iter, model, num_class,word2idx,tag2idx,device):
    """
    train hmm and evaluate
    """
    