from gnninterpreter import *
import torch
from tqdm.auto import trange


reddit = Redditdataset(seed=12345)

reddit_train, reddit_test = reddit.train_test_split(k=10)
reddit_model = GCNClassifier(node_features=1,    #Watch out, i changed this
                            num_classes=2,      #Watch out, i changed this
                            hidden_channels=64,
                            num_layers=5)

for epoch in trange(30):
    train_loss = reddit_train.fit_model(reddit_model, lr=0.001)
    train_f1 = reddit_train.evaluate_model(reddit_model)
    val_f1 = reddit_test.evaluate_model(reddit_model)
    print(f'Epoch: {epoch:03d}, '
          f'Train Loss: {train_loss:.4f}, '
          f'Train F1: {train_f1}, '
          f'Test F1: {val_f1}')

torch.save(reddit_model.state_dict(), 'ckpts/reddit.pt')
