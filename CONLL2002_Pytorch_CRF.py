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
from data_preparation2 import ConLL2002DataSet

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
from data_preparation2 import Vectorizer

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

# (batch_size, seq_len, 1) --> (batch_size, seq_len, tag_dim)
# the dimension = 1 is the one giving the token indexes
from models import LSTMTagger

# (batch_size, seq_len, tag_dim) --> (batch_size, seq_len, 1), selecting the best tag
# sequence using Viterbi decoding
# import from NCRF++
# from ncrfpp import CRF
from pytorchcrf import CRF


class LSTM_CRFTagger(nn.Module):
    # based on SeqLabel from NCRF++
    # This is a wrapper to use the CRF after LSTMtagger

    def __init__(self, lstm_args, gpu=False):
        super(LSTM_CRFTagger, self).__init__()
        ## add two more labels for downlayer lstm, use original label size for CRF
        tagset_size = lstm_args["tagset_size"]
        # lstm_args["tagset_size"] += 2
        self.lstm = LSTMTagger(**lstm_args)
        # self.crf = CRF(tagset_size, gpu)
        self.crf = CRF(tagset_size, batch_first=True)

    @staticmethod
    def _get_mask(X_lens, batch_size, seq_len):
        mask = Variable(torch.zeros((batch_size, seq_len))).byte()
        for idx, X_len in enumerate(X_lens):
            mask[idx, :X_len] = torch.Tensor([1]*int(X_len))
        return mask

    def loss(self, tag_scores, mask, y_hat):
        # ncrf++
        # total_loss = self.crf.neg_log_likelihood_loss(tag_scores, mask, y_hat)
        total_loss = self.crf.forward(tag_scores, y_hat, mask)
        # normalise by batch_size
        return total_loss / tag_scores.size(0)

    def forward(self, X, X_lens, nbest=None):
        tag_scores = self.lstm(X, X_lens)
        batch_size, seq_len = X.size()
        mask = self._get_mask(X_lens, batch_size, seq_len)
        # use this for training
        if not nbest:
            return tag_scores, mask
        # use this for testing
        # TODO: check speed of _viterbi_decode_nbest vs _viterbi_decode. If nbest=1
        # maybe it's better to use _viterbi_decode
        else:
            # scores, tag_seq = self.crf._viterbi_decode_nbest(tag_scores, mask, nbest)
            # return scores, tag_seq
            return self.crf.decode(tag_scores, mask)



from evaluation import eval_model_for_set

# ## Set hyperparams and train the model

BATCH_SIZE = 32
EPOCHS = 50
PRINT_EVERY_NBATCHES = 100
PRINT_EVERY_NEPOCHS = 1
lstm_args = {
    "embedding_dim": 30,
    "hidden_dim": 10,
    "vocab_size": len(vectorizer.word_vocab),
    "tagset_size": len(vectorizer.tag_vocab),
    "bidirectional": False
}
model = LSTM_CRFTagger(lstm_args)
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
        tag_scores, mask = model(batch_sents, batch_lens)
        # Step 3. Compute the loss, gradients, and update the parameters
        loss = model.loss(tag_scores, mask, batch_tags)
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
        train_loss.append(eval_model_for_set(model, train_data, vectorizer, True))
        logger.info(f"Loss: {train_loss[-1]}")
        logger.info("**********VALIDATION PERFORMANCE*********")
        val_loss.append(eval_model_for_set(model, val_data, vectorizer, True))
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
# * Use trained embeddings / hand-crafted features
# * CNN
#
# Speed:
# * _viterbi_decode_nbest vs _viterbi_decode when nbest=1
#
# Coding:
# * Use `DataLoader` from Pytorch rather than `batches_generator`

