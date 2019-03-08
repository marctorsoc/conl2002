# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python (marc)
#     language: python
#     name: marc
# ---

# %%time
import nltk
import pandas as pd
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))


# ## $(token, pos, tag)^N$ --> $(tokens, tags)$

def get_tokens_tags_from_sents(sents):
    tokens, tags = [], []
    for sent in sents:
        sent_tokens, _, sent_tags = list(zip(*sent))
        tokens.append(sent_tokens)
        tags.append(sent_tags)
    return tokens, tags


# %%time
train_tokens, train_tags = get_tokens_tags_from_sents(train_sents)
val_tokens, val_tags = get_tokens_tags_from_sents(test_sents)

# You should always understand what kind of data you deal with. For this purpose, you can print the data running the following cell:

idx = 0
pd.DataFrame([train_tokens[idx], train_tags[idx]])

# ### Prepare mappings
#
# To train a neural network, we will use two mappings: 
# - {token}$\to${token id}: address the row in embeddings matrix for the current token;
# - {tag}$\to${tag id}: one-hot ground truth probability distribution vectors for computing the loss at the output of the network.
#
# Now you need to implement the function *build_dict* which will return {token or tag}$\to${index} and vice versa. 

from collections import defaultdict


def build_dict(tokens_or_tags, special_tokens):
    # Create a dictionary with default value 0
    tok2idx = defaultdict(lambda: 0)
    ind = 0
    for t in special_tokens:
        tok2idx[t] = ind
        ind += 1
    for sam in tokens_or_tags:
        for t in sam:
            if t not in special_tokens and t not in tok2idx:
                tok2idx[t] = ind
                ind += 1
    return tok2idx, dict((v, k) for k, v in tok2idx.items())


# After implementing the function *build_dict* you can make dictionaries for tokens and tags. Special tokens for tokens will be:
#  - `<UNK>` token for out of vocabulary tokens; index = 0
#  - `<PAD>` token for padding sentence to the same length when we create batches of sentences. index = 1

# +
special_tokens = ['<UNK>', '<PAD>']
special_tags = ['O']

# Create dictionaries 
token2idx, idx2token = build_dict(train_tokens, special_tokens)
tag2idx, idx2tag = build_dict(train_tags, special_tags)


# -

# ### Generate batches
#
# Neural Networks are usually trained with batches. It means that weight updates of the network are based on several sequences at every single time. The tricky part is that all sequences within a batch need to have the same length. So we will pad them with a special `<PAD>` token. It is also a good practice to provide RNN with sequence lengths, so it can skip computations for padding parts. We provide the batching function *batches_generator* readily available for you to save time. 

def prepare_sequence(seq, to_ix):
    def to_ix_defaults(t):
        return to_ix.get(t, to_ix.get("<UNK>", to_ix['O']))
    return torch.tensor(np.vectorize(to_ix_defaults)(seq), dtype=torch.long)


def batches_generator(batch_size, tokens, tags,
                      shuffle=True, allow_smaller_last_batch=True, seed=8):
    """Generates padded batches of tokens and tags."""
    # TODO: use DataLoader from Pytorch for this
    # tokens is a list of docs, and each docs is a list of tokens
    # SHUFFLE
    n_samples = len(tokens)
    np.random.seed(seed)
    if shuffle:
        order = np.random.permutation(n_samples)
    else:
        order = np.arange(n_samples)

    # NUMBER OF BATCHES
    n_batches = n_samples // batch_size
    # and n_samples / batch_size not integer, put the leftovers in last batch
    if allow_smaller_last_batch and n_samples % batch_size:
        n_batches += 1

    # for each batch, get the docs, labels and real lengths and yield them
    for k in range(n_batches):
        batch_start = k * batch_size
        batch_end = min((k + 1) * batch_size, n_samples)
        x, y = [], []
        batch_lengths = torch.zeros(batch_end - batch_start, dtype=torch.int32)
        # x will be a list of lists of indices (one list of indices per doc in this batch)
        for sample_in_batch_index, sample_idx in enumerate(order[batch_start: batch_end]):
            try:
                x.append(prepare_sequence(tokens[sample_idx], token2idx))
                y.append(prepare_sequence(tags[sample_idx], tag2idx))
            except Exception as marc:
                print(marc)
                import pdb; pdb.set_trace()
            batch_lengths[sample_in_batch_index] = len(tags[sample_idx])
        x = pad_sequence(x, batch_first=True, padding_value=token2idx["<PAD>"])
        y = pad_sequence(y, batch_first=True, padding_value=-1)
        batch_lengths, perm_idx = batch_lengths.sort(0, descending=True)
        # yield each batch
        yield x[perm_idx, ...], y[perm_idx, ...], batch_lengths


# ### Model

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import (
    pack_padded_sequence, pad_sequence, pad_packed_sequence
)
from torch.nn.utils import clip_grad_norm_
torch.manual_seed(1)
import numpy as np
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score


class LSTMTagger(nn.Module):

    def __init__(self, batch_size, embedding_dim, hidden_dim, vocab_size, tagset_size,
                padding_idx, verbose=False, bidirectional=False):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                           bidirectional=bidirectional)
        self.tagset_size = tagset_size
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear((1+bidirectional)*hidden_dim, tagset_size)
        self.verbose = verbose

    def forward(self, X, X_lens):
        # embeddings
        embeds = self.word_embeddings(X)
        if self.verbose: print(f"Embeds: {embeds.size()}")
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        embeds = pack_padded_sequence(embeds, X_lens.cpu().numpy(), batch_first=True)
        # lstm
        #lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = self.lstm(embeds)
        # undo the packing operation
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        if self.verbose: print(f"lstm_out: {lstm_out.size()}")
        # (batch_size, seq_len, hidden_dim) --> (batch_size * seq_len, hidden_dim)
        s = lstm_out.contiguous().view(-1, lstm_out.shape[2])
        # (batch_size * seq_len, hidden_dim) --> (batch_size * seq_len, tag_dim)
        tag_space = self.hidden2tag(lstm_out)
        if self.verbose: print(f"tag space: {tag_space.size()}")
        # normalize logits
        tag_scores = F.log_softmax(tag_space, dim=1)
        if self.verbose: print(f"tag scores: {tag_scores.size()}")
        return tag_scores
    
    def loss(self, Y_hat, Y):
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        return criterion(Y_hat.view(-1, Y_hat.size()[2]), Y.view(-1))


# ### Evaluation helpers

labels_to_score = list(tag2idx.keys())
labels_to_score.remove('O')
labels_to_score

# group B and I results
sorted_labels = sorted(
    labels_to_score, 
    key=lambda name: (name[1:], name[0])
)


# +
def predict_tags(model, batch_tokens, batch_lengths, batch_tags=None):
    """Performs predictions and transforms indices to tokens and tags."""
    
    tag_scores = model(batch_tokens, batch_lengths)
    predicted_tags = np.vectorize(idx2tag.get)(torch.argmax(tag_scores, dim=2).data.numpy())
    if batch_tags is not None:
        return predicted_tags, model.loss(tag_scores, batch_tags)
    return predicted_tags


def my_scorer(true_tags, predicted_tags):
    score = flat_f1_score(true_tags, predicted_tags, average='weighted', labels=sorted_labels)
    logger.info(f"f1 score: {score:.3f}")
    print(flat_classification_report(
        true_tags, predicted_tags, labels=sorted_labels, digits=3
    ))


def eval_model_for_set(model, tokens, tags, scoring=my_scorer):
    """Computes NER quality measures given model and a dataset"""
    model.eval()
    predicted_tags, true_tags, loss = [], [], 0
    with torch.no_grad():
        for x_batch, y_batch, lengths in batches_generator(len(tokens), tokens, tags):
            padded_predicted_tags, batch_loss = predict_tags(model, x_batch, lengths, y_batch)
            loss += batch_loss
            padded_true_tags = np.vectorize(idx2tag.get)(y_batch.data)
            for x, y, l in zip(padded_predicted_tags, padded_true_tags, lengths): 
                predicted_tags.append(x[:l])
                true_tags.append(y[:l])    
        scoring(true_tags, predicted_tags)
        return loss / len(true_tags)


# -

# ## Set hyperparams and train the model

EMBEDDING_DIM = 200
HIDDEN_DIM = 200
BATCH_SIZE = 32
EPOCHS = 50
VOCAB_SIZE = len(token2idx)
TAGSET_SIZE = len(tag2idx)
PADDING_IDX = token2idx["<PAD>"]
training_data = (train_tokens, train_tags)
model = LSTMTagger(BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, 
                   TAGSET_SIZE, PADDING_IDX, verbose=False, bidirectional=True)
optimiser = torch.optim.Adam(model.parameters(), lr=0.005)

# +
# %%time
# print predictions before training
#print_example(training_data, 123, model, token2idx, idx2tag)
logger.info("START!")
train_loss, val_loss = [], []
for epoch in range(EPOCHS): 
    train_loader = batches_generator(BATCH_SIZE, train_tokens, train_tags, seed=epoch)
    epoch_loss = 0
    model.train()
    for idx_batch, batch in enumerate(train_loader):
        batch_sents, batch_tags, batch_lens = batch
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        # Step 2. Run our forward pass.
        tag_scores = model(batch_sents, batch_lens)
        # Step 3. Compute the loss, gradients, and update the parameters
        loss = model.loss(tag_scores, batch_tags)
        loss.backward()
        epoch_loss += float(loss)
        clip_grad_norm_(model.parameters(), 5)
        optimiser.step()
        # disabled for now
        if (idx_batch + 1) % 970 == 0:
            logger.info(
                f'Epoch [{epoch + 1}/{EPOCHS}], '
                f"Step [{idx_batch + 1}/{len(train_tags)// BATCH_SIZE}], "
                f"Loss: {loss:.4f}"
            )
        
    logger.info(f"avg epoch {epoch + 1} train loss: {epoch_loss/(idx_batch + 1):.4f}")
    if ((epoch + 1) % 5) == 0:
        logger.info("**********TRAINING PERFORMANCE*********")
        train_loss.append(eval_model_for_set(model, train_tokens, train_tags))
        logger.info(f"Loss: {train_loss[-1]}")
        logger.info("**********VALIDATION PERFORMANCE*********")
        val_loss.append(eval_model_for_set(model, val_tokens, val_tags))
        logger.info(f"Loss: {val_loss[-1]}")

# print predictions after training
#print_example(training_data, 123, model, token2idx, idx2tag)
#print(training_data[1][123])
# -

# ### Ideas to improve

# Accuracy:
# * Dropout
# * Early stopping
# * Fine-tunning hyperparams: learning rate (https://www.jeremyjordan.me/nn-learning-rate/), embedding and hidden dimensions
# * Use trained embeddings
# * CRF / CNN
#
# Coding:
# * Use `DataLoader` from Pytorch rather than `batches_generator`


def print_example(training_data, i, model, word2idx, idx2tag):
    pass
    # Note that element i,j of tag_scores is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    # with torch.no_grad():
    #     seq = training_data[0][i]
    #     labs = training_data[1][i]
    #     inputs = prepare_sequence(seq, word2idx)
    #     tag_scores = model(inputs.view(1, len(inputs)),
    #                        torch.tensor([len(seq)]))
    #     tags = np.vectorize(idx2tag.get)(torch.argmax(tag_scores, dim=2).data.numpy())
    #     print(seq)
    #     print()
    #     print(tags)
    #     print()
    #     print(len(seq), tag_scores.size(), tags.shape)
    #     print()
    #     print(training_data[1][i])
    #     print(training_data[1][i] == tags)
#print_example(training_data, 79, model, token2idx, idx2tag)
