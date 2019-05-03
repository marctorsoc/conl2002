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

# You should always understand what kind of data you deal with. For this purpose, you
# can print the data running the following cell:

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
from torch import optim
from torch.nn.utils import clip_grad_norm_

torch.manual_seed(1)

# (batch_size, seq_len, 1) --> (batch_size, seq_len, tag_dim)
# the dimension = 1 is the one giving the token indexes
from models import LstmTagger

# (batch_size, seq_len, tag_dim) --> (batch_size, seq_len, 1), selecting the best tag
# sequence using Viterbi decoding
from torch.autograd import Variable
from allennlp.modules import ConditionalRandomField


class LSTM_CRFTagger(nn.Module):
    # based on CRFtagger from AllenNLP
    # This is a wrapper to use the CRF after LSTMtagger, so the flow is:
    # embedding -- lstm -- hidden2tag (dense layer) -- CRF

    def __init__(self, lstm_args):
        super(LSTM_CRFTagger, self).__init__()
        self.lstm = LstmTagger(**lstm_args)
        self.crf = ConditionalRandomField(
            lstm_args["tagset_size"], include_start_end_transitions=True
        )

    @staticmethod
    def _get_mask(X_lens, batch_size, seq_len):
        mask = Variable(torch.zeros((batch_size, seq_len))).byte()
        for idx, X_len in enumerate(X_lens):
            mask[idx, :X_len] = torch.ones(X_len)
        return mask

    def forward(self, input, input_lens):
        logits = self.lstm.forward(input, input_lens, apply_softmax=False)
        batch_size, seq_len, _ = logits.size()
        mask = __class__._get_mask(input_lens, batch_size, seq_len)
        return logits, mask

    def loss(self, logits, mask, target):
        """Use negative log-likelihood as loss"""
        log_likelihood = self.crf(logits, target, mask)
        return -log_likelihood

    def decode(self, logits, mask):
        """Return most probable sequence using Viterbi"""
        best_paths = self.crf.viterbi_tags(logits, mask)
        # Just get the tags and ignore the score
        return [best_sequence for best_sequence, score in best_paths]


from evaluation import eval_model_for_set
from torch.utils.data import DataLoader
from data_preparation import pad_and_sort_batch


# ## Set hyperparams and train the model

BATCH_SIZE = 32
EPOCHS = 5
PRINT_EVERY_NBATCHES = 100
PRINT_EVERY_NEPOCHS = 1
lstm_args = {
    "embedding_dim": 200,
    "hidden_dim": 200,
    "vocab_size": len(vectorizer.word_vocab),
    "tagset_size": len(vectorizer.tag_vocab),
    "bidirectional": True
}
model = LSTM_CRFTagger(lstm_args)
LEARNING_RATE = 0.005
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
# * Clean NCRF++ implementation, probably more efficient
