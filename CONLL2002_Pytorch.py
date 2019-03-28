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

import pandas as pd
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


# ## $(token, pos, tag)^N$ --> $(tokens, tags)$

# %%time
from data_preparation import ConLL2002DataSet

train_tokens, train_tags = ConLL2002DataSet("esp.train").get_tokens_tags_from_sents()
val_tokens, val_tags = ConLL2002DataSet("esp.testb").get_tokens_tags_from_sents()

# You should always understand what kind of data you deal with. For this purpose, you can print the data running the following cell:

idx = 0
pd.DataFrame([train_tokens[idx], train_tags[idx]])

# ### Prepare mappings
#
# A neural network needs to work with word indices, not next. Then, we need to learn
# the vocabulary of tokens and tags. This is accomplished with the Vectorizer, and then
# used to transform the datasets into VectorizedDataset objects
#
# Some special tokens in the vocabulary:
#  - `<PAD>` token for padding sentence to the same length when we create batches of
#  sentences. index = 0
#  - `<UNK>` token for out of vocabulary tokens; index = 1
#  - `<START>` index = 2 (not used here)
#  - `<END>` index = 3 (not used here)

# +
from data_preparation import Vectorizer

vectorizer = Vectorizer(use_start_end=False, use_pad=True)
vectorizer.fit(train_tokens, train_tags)
train_data = vectorizer.transform(train_tokens, train_tags)
val_data = vectorizer.transform(val_tokens, val_tags)
# -

print(train_tokens[0])
print(train_data.input[0])
vectorizer.map_sequence_back(vectorizer.word_vocab, train_data.input[0])

print(train_tags[0])
print(train_data.target[0])
vectorizer.map_sequence_back(vectorizer.tag_vocab, train_data.target[0])

# ### Generate batches
#
# Neural Networks are usually trained with batches. It means that weight
# updates of the network are based on several sequences at every single time.
# The tricky part is that all sequences within a batch need to have the same
# length. So we will pad them with a special `<PAD>` token. It is also a good
# practice to provide RNN with sequence lengths, so it can skip computations
# for padding parts. We provide the batching function *batches_generator*
# readily available for you to save time.

# ### Model

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import clip_grad_norm_

torch.manual_seed(1)


class LSTMTagger(nn.Module):
    """
    This class will define the following three blocks:
    1. Embedding layer: from word index to embedding
    2. (Bi)LSTM: from embedding to a representation of dimension hidden_dim
    3. Hidden2tag: a dense layer from hidden_dim to the tag space
    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        vocab_size,
        tagset_size,
        verbose=False,
        bidirectional=False,
    ):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
        )
        self.tagset_size = tagset_size
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear((1 + bidirectional) * hidden_dim, tagset_size)
        self.verbose = verbose

    def forward(self, X, X_lens):
        # embeddings
        embeds = self.word_embeddings(X)
        if self.verbose:
            print(f"Embeds: {embeds.size()}")
        # pack_padded_sequence so that padded items in the sequence won't be
        # shown to the LSTM
        embeds = pack_padded_sequence(embeds, X_lens.cpu().numpy(), batch_first=True)
        # lstm
        # lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = self.lstm(embeds)
        # undo the packing operation
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        if self.verbose:
            print(f"lstm_out: {lstm_out.size()}")
        #  (batch_size * seq_len, hidden_dim) --> (batch_size * seq_len, tag_dim)
        tag_space = self.hidden2tag(lstm_out)
        if self.verbose:
            print(f"tag space: {tag_space.size()}")
        # normalize logits
        tag_scores = F.log_softmax(tag_space, dim=1)
        if self.verbose:
            print(f"tag scores: {tag_scores.size()}")
        return tag_scores

    def loss(self, y_hat, y):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        return criterion(y_hat.view(-1, y_hat.size()[2]), y.view(-1))


from evaluation import eval_model_for_set
from torch.utils.data import DataLoader
from data_preparation import pad_and_sort_batch

# ## Set hyperparams and train the model

EMBEDDING_DIM = 100
HIDDEN_DIM = 100
BATCH_SIZE = 32
EPOCHS = 5
VOCAB_SIZE = len(vectorizer.word_vocab)
TAGSET_SIZE = len(vectorizer.tag_vocab)
PADDING_IDX = 0
PRINT_EVERY_NBATCHES = 100
PRINT_EVERY_NEPOCHS = 1
model = LSTMTagger(
    EMBEDDING_DIM,
    HIDDEN_DIM,
    VOCAB_SIZE,
    TAGSET_SIZE,
    verbose=False,
    bidirectional=True,
)
LEARNING_RATE = 0.005
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# +
# %%time
logger.info("START!")
train_loss, val_loss = [], []
for epoch in range(EPOCHS):
    # TODO: review how to set the seed
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_and_sort_batch
    )
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
        if (idx_batch + 1) % PRINT_EVERY_NBATCHES == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{EPOCHS}], "
                f"Step [{idx_batch + 1}/{len(train_tags)// BATCH_SIZE}], "
                f"Loss: {loss:.4f}"
            )

    logger.info(f"avg epoch {epoch + 1} train loss: {epoch_loss/(idx_batch + 1):.4f}")
    if ((epoch + 1) % PRINT_EVERY_NEPOCHS) == 0:
        logger.info("**********TRAINING PERFORMANCE*********")
        train_loss.append(eval_model_for_set(model, train_data, vectorizer))
        logger.info(f"Loss: {train_loss[-1]}")
        logger.info("**********VALIDATION PERFORMANCE*********")
        val_loss.append(eval_model_for_set(model, val_data, vectorizer))
        logger.info(f"Loss: {val_loss[-1]}")

# print predictions after training
# print_example(training_data, 123, model, token2idx, idx2tag)
# print(training_data[1][123])
# -

# ### Conclusions

# Really decent, given the simplicity of the model (it's just a BiLSTM with a dense layer afterwards). Lot of overfitting

# ### Ideas to improve

# Accuracy:
# * Dropout
# * Early stopping
# * Fine-tunning hyperparams: learning rate (https://www.jeremyjordan.me/nn-learning-rate/), embedding and hidden dimensions
# * Use trained embeddings
# * CRF / CNN
