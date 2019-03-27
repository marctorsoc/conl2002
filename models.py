from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class LSTMTagger(nn.Module):
    """
    This class defines the following three blocks:
    1. Embedding layer: from word index to embedding
    2. (Bi)LSTM: from embedding to a representation of dimension hidden_dim
    3. Hidden2tag: a dense layer from hidden_dim to the tag space
    """

    def __init__(
        self, embedding_dim, hidden_dim, vocab_size, tagset_size, bidirectional=False
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

    def forward(self, X, X_lens):
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
        tag_space = self.hidden2tag(lstm_out)
        # normalize logits
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def loss(self, y_hat, y):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        return criterion(y_hat.view(-1, y_hat.size()[2]), y.view(-1))
