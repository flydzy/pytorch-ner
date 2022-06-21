from preprocess import preprocess

train_iter = preprocess.build_iterator('data/en/eng.train', batch_size=32)
for batch in train_iter:
    print(batch.dataset.examples[0].sent)