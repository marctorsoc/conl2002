import torch
from allennlp.modules import ConditionalRandomField
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import sys
if sys.platform == "linux":
    sys.path.append("/home/ubuntu/NCRFpp")
else:
    sys.path.append("/Users/marc.torrellas/HackDay/NCRFpp")
from model.crf import CRF as NCRFpp_CRF


class LstmTagger(nn.Module):
    """
    This class defines the following three blocks:
    1. Embedding layer: from word index to embedding
    2. (Bi)LSTM: from embedding to a representation of dimension hidden_dim
    3. Hidden2tag: a dense layer from hidden_dim to the tag space
    """

    def __init__(
        self, embedding_dim, hidden_dim, vocab_size, tagset_size, bidirectional=False
    ):
        super(LstmTagger, self).__init__()
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

    def forward(self, X, X_lens, apply_softmax=True):
        # embeddings
        embeds = self.word_embeddings(X)
        # pack_padded_sequence so that padded items in the sequence won't be
        # shown to the LSTM
        embeds = pack_padded_sequence(embeds, X_lens.cpu().numpy(), batch_first=True)
        # lstm
        lstm_out, _ = self.lstm(embeds)
        # undo the packing operation
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # (batch_size, seq_len, hidden_dim) --> (batch_size * seq_len, tag_dim)
        tag_scores = self.hidden2tag(lstm_out)
        # normalize logits
        if apply_softmax:
            tag_scores = F.log_softmax(tag_scores, dim=1)
        return tag_scores

    def loss(self, y_hat, y):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        return criterion(y_hat.view(-1, y_hat.size()[2]), y.view(-1))


class CrfTagger(nn.Module):
    # based on CRFtagger from AllenNLP
    # This is a wrapper to use the CRF after LSTMtagger, so the flow is:
    # embedding -- lstm -- hidden2tag (dense layer) -- CRF

    def __init__(self, kwargs):
        super(CrfTagger, self).__init__()
        self.gpu = kwargs.pop("use_gpu", False)
        if kwargs.pop("use_lstm", False):
            self.lstm = LstmTagger(**kwargs)
        self.crf = ConditionalRandomField(
            kwargs["tagset_size"], include_start_end_transitions=True
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


class CrfTagger2(nn.Module):
    # based on SeqLabel in NCRFpp
    def __init__(self, kwargs):
        super(CrfTagger2, self).__init__()
        self.gpu = kwargs.pop("use_gpu", False)
        self.average_batch = kwargs.pop("average_batch", True)
        self.crf = NCRFpp_CRF(kwargs["tagset_size"], self.gpu)
        if kwargs.pop("use_lstm", False):
            kwargs["tagset_size"] += 2
            self.lstm = LstmTagger(**kwargs)


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
        total_loss = self.crf.neg_log_likelihood_loss(logits, mask, target)
        batch_size, seq_len, _ = logits.size()
        if self.average_batch:
            total_loss = total_loss / batch_size
        return total_loss

    def decode(self, logits, mask, return_scores=False):
        scores, tag_seq = self.crf.viterbi_decode(logits, mask)
        if return_scores:
            return scores, tag_seq
        return tag_seq

    def decode_nbest(self, logits, mask, nbest, return_scores=False):
        scores, tag_seq = self.crf.viterbi_decode_nbest(logits, mask, nbest)
        if return_scores:
            return scores, tag_seq
        return tag_seq
