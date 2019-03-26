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

import nltk
import pandas as pd
import logging


# %%time
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))


# ## $(token, pos, tag)^N$ --> $(tokens, tags)$

# %%time
from data_preparation import get_tokens_tags_from_sents
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


# After implementing the function *build_dict* you can make dictionaries for tokens and tags. Special tokens will be:
#  - `<UNK>` token for out of vocabulary tokens; index = 0
#  - `O` tag for no entity OR unknown tag. index = 0

# +
special_tokens = ['<UNK>']
special_tags = ['O']

# Create dictionaries
from data_preparation import build_dict
token2idx, idx2token = build_dict(train_tokens, special_tokens)
tag2idx, idx2tag = build_dict(train_tags, special_tags)


# -

# ### Generate batches
#
# Neural Networks are usually trained with batches. It means that weight
# updates of the network are based on several sequences at every single time.
# The tricky part is that all sequences within a batch need to have the same
# length. So we will pad them with a special `<PAD>` token. It is also a good
# practice to provide RNN with sequence lengths, so it can skip computations
# for padding parts. We provide the batching function *batches_generator*
# readily available for you to save time.

from data_preparation import batches_generator


# ### Model

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import (
    pack_padded_sequence, pad_packed_sequence
)
from torch.autograd import Variable
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


# class CRF(nn.Module):
#
#     START_TAG = -1
#     STOP_TAG = -2
#
#     def __init__(self, tagset_size, gpu):
#         super(CRF, self).__init__()
#         # Matrix of transition parameters.  Entry i,j is the score of
#         # transitioning from i to j.
#         self.tagset_size = tagset_size
#         # # We add 2 here, because of START_TAG and STOP_TAG
#         # # transitions (f_tag_size, t_tag_size), transition value
#         # from f_tag to t_tag
#         init_transitions = torch.zeros(self.tagset_size + 2,
#                                        self.tagset_size + 2)
#         init_transitions[:, self.START_TAG] = -10000.0
#         init_transitions[self.STOP_TAG, :] = -10000.0
#         init_transitions[:, 0] = -10000.0
#         init_transitions[0, :] = -10000.0
#         self.transitions = nn.Parameter(init_transitions)
#
#     @staticmethod
#     def _argmax(vec):
#         # return the argmax as a python int
#         _, idx = torch.max(vec, 1)
#         return idx.item()
#
#     # Compute log sum exp in a numerically stable way for the forward algorithm
#     def _log_sum_exp(self, vec):
#         max_score = vec[0, self._argmax(vec)]
#         max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
#         return max_score + \
#                torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
#
#     def _forward_alg(self, feats):
#         # Do the forward algorithm to compute the partition function
#         init_alphas = torch.full((1, self.tagset_size), -10000.)
#         # START_TAG has all of the score.
#         init_alphas[0][self.START_TAG] = 0.
#
#         # Wrap in a variable so that we will get automatic backprop
#         forward_var = init_alphas
#
#         # Iterate through the sentence
#         for feat in feats:
#             alphas_t = []  # The forward tensors at this timestep
#             for next_tag in range(self.tagset_size):
#                 # broadcast the emission score: it is the same regardless of
#                 # the previous tag
#                 emit_score = feat[next_tag].view(
#                     1, -1).expand(1, self.tagset_size)
#                 # the ith entry of trans_score is the score of transitioning to
#                 # next_tag from i
#                 trans_score = self.transitions[next_tag].view(1, -1)
#                 # The ith entry of next_tag_var is the value for the
#                 # edge (i -> next_tag) before we do log-sum-exp
#                 next_tag_var = forward_var + trans_score + emit_score
#                 # The forward variable for this tag is log-sum-exp of all the
#                 # scores.
#                 alphas_t.append(self._log_sum_exp(next_tag_var).view(1))
#             forward_var = torch.cat(alphas_t).view(1, -1)
#         terminal_var = forward_var + self.transitions[self.STOP_TAG]
#         alpha = self._log_sum_exp(terminal_var)
#         return alpha
#
#     def _score_sentence(self, feats, tags):
#         # Gives the score of a provided tag sequence
#         score = torch.zeros(1)
#         tags = torch.cat(
#             [torch.tensor([self.START_TAG], dtype=torch.long), tags])
#         for i, feat in enumerate(feats):
#             score = score + \
#                     self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
#         score = score + self.transitions[self.STOP_TAG, tags[-1]]
#         return score
#
#     def _viterbi_decode(self, feats):
#         backpointers = []
#
#         # Initialize the viterbi variables in log space
#         init_vvars = torch.full((1, self.tagset_size), -10000.)
#         init_vvars[0][self.START_TAG] = 0
#
#         # forward_var at step i holds the viterbi variables for step i-1
#         forward_var = init_vvars
#         for feat in feats:
#             bptrs_t = []  # holds the backpointers for this step
#             viterbivars_t = []  # holds the viterbi variables for this step
#
#             for next_tag in range(self.tagset_size):
#                 # next_tag_var[i] holds the viterbi variable for tag i at the
#                 # previous step, plus the score of transitioning
#                 # from tag i to next_tag.
#                 # We don't include the emission scores here because the max
#                 # does not depend on them (we add them in below)
#                 next_tag_var = forward_var + self.transitions[next_tag]
#                 best_tag_id = self._argmax(next_tag_var)
#                 bptrs_t.append(best_tag_id)
#                 viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
#             # Now add in the emission scores, and assign forward_var to the set
#             # of viterbi variables we just computed
#             forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
#             backpointers.append(bptrs_t)
#
#         # Transition to STOP_TAG
#         terminal_var = forward_var + self.transitions[self.STOP_TAG]
#         best_tag_id = self._argmax(terminal_var)
#         path_score = terminal_var[0][best_tag_id]
#
#         # Follow the back pointers to decode the best path.
#         best_path = [best_tag_id]
#         for bptrs_t in reversed(backpointers):
#             best_tag_id = bptrs_t[best_tag_id]
#             best_path.append(best_tag_id)
#         # Pop off the start tag (we dont want to return that to the caller)
#         start = best_path.pop()
#         assert start == self.START_TAG  # Sanity check
#         best_path.reverse()
#         return path_score, best_path
#
#     def neg_log_likelihood(self, lstm_feats, tags):
#         forward_score = self._forward_alg(lstm_feats)
#         gold_score = self._score_sentence(lstm_feats, tags)
#         return forward_score - gold_score
#
#     def forward(self, lstm_feats):  # dont confuse this with _forward_alg above.
#         # Find the best path, given the emission scores (here the features
#         # obtained from the BiLSTM)
#         score, tag_seq = self._viterbi_decode(lstm_feats)
#         return score, tag_seq
#
#
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


# ### Evaluation helpers

labels_to_score = list(tag2idx.keys())
labels_to_score.remove('O')
labels_to_score

# group B and I results
sorted_labels = sorted(
    labels_to_score,
    key=lambda name: (name[1:], name[0])
)


from evaluation import eval_model_for_set

# ## Set hyperparams and train the model

BATCH_SIZE = 32
EPOCHS = 50
PRINT_EVERY_NBATCHES = 100
PRINT_EVERY_NEPOCHS = 1
lstm_args = {
    "embedding_dim": 30,
    "hidden_dim": 10,
    "vocab_size": len(token2idx), 
    "tagset_size": len(tag2idx), 
    "padding_idx": -1,
    "verbose": False, 
    "bidirectional": False
}
model = LSTM_CRFTagger(lstm_args)
LEARNING_RATE = 0.005
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# +
# %%time
# print predictions before training
#print_example(training_data, 123, model, token2idx, idx2tag)
logger.info("START!")
train_loss, val_loss = [], []
for epoch in range(EPOCHS):
    train_loader = batches_generator(
        BATCH_SIZE, train_tokens, train_tags, token2idx, tag2idx, seed=epoch
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
                f'Epoch [{epoch + 1}/{EPOCHS}], '
                f"Step [{idx_batch + 1}/{len(train_tags)// BATCH_SIZE}], "
                f"Loss: {loss:.4f}"
            )

    logger.info(f"avg epoch {epoch + 1} train loss: {epoch_loss/(idx_batch + 1):.4f}")
    if ((epoch + 1) % PRINT_EVERY_NEPOCHS) == 0:
        logger.info("**********TRAINING PERFORMANCE*********")
        train_loss.append(eval_model_for_set(
            model, train_tokens, train_tags, token2idx, tag2idx, sorted_labels, True
        ))
        logger.info(f"Loss: {train_loss[-1]}")
        logger.info("**********VALIDATION PERFORMANCE*********")
        val_loss.append(eval_model_for_set(
            model, val_tokens, val_tags, token2idx, tag2idx, sorted_labels, True
        ))
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
