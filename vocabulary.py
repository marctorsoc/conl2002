from collections import Counter
import six


#
# Modified implementation based on the one in
# https://github.com/joosthub/pytorch-nlp-tutorial-eu2018/


class Vocabulary(object):
    """
    An implementation that manages the interface between a token dataset and the
        machine learning algorithm.
    """

    def __init__(self, use_unks=False, unk_token="<UNK>",
                 use_pad=False, pad_token="<PAD>", use_start_end=False,
                 start_token="<START>", end_token="<END>"):
        """
        Args:
            use_unks (bool): The vocabulary will output UNK tokens for out of
                vocabulary items.
                [default=False]
            unk_token (str): The token used for unknown tokens.
                If `use_unk` is True, this will be added to the vocabulary.
                [default='<UNK>']
            use_pad (bool): The vocabulary will reserve the 0th index for a
                padding token.
                This is used to handle variable lengths in sequence models.
                [default=False]
            pad_token (str): The token used for the padding.
                Note: mostly a placeholder; it's unlikely the token will be seen
                [default='<PAD>']
            use_start_end (bool): The vocabulary will reserve indices for two
                tokens that represent the start and end of a sequence.
                [default=False]
            start_token: The token used to indicate the start of a sequence.
                If `use_start_end` is True, this will be added to the vocabulary
                [default='<START>']
            end_token: The token used to indicate the end of a sequence
                If `use_start_end` is True, this will be added to the vocabulary
                [default='<END>']
        """

        self._token2idx = {}  # str -> integer index
        self._idx2token = {}  # integer index -> str;
        self._counts = Counter()  # int -> int; count occurrences
        # A set of indexes for which UNK is returned. Used e.g. for tokens
        # appearing less than 5 times
        self._forced_unks = set()
        self._next_index = 0
        self._frozen = False  # used to avoid the vocabulary to keep growing
        # minimum count to include a token in the vocabulary, 0 to disable
        self._frequency_threshold = 0

        # pad token for use in masked recurrent networks
        # usually need to be the 0th index
        self.use_pad = use_pad
        self.pad_token = pad_token
        if self.use_pad:
            self[self.pad_token]

        # unk token for out of vocabulary (OOV) tokens, will be 1st index
        self.use_unk = use_unks
        self.unk_token = unk_token
        if self.use_unk:
            self[self.unk_token]

        # start token for sequence models, will be 2nd and 3rd indexes
        self.use_start_end = use_start_end
        self.start_token = start_token
        self.end_token = end_token
        if self.use_start_end:
            self[self.start_token]
            self[self.end_token]

    def __contains__(self, k):
        return k in self._token2idx

    def __len__(self):
        return len(self._token2idx)

    def __getitem__(self, key):
        """ interface to use and fit token2idx"""
        # test case
        if self._frozen:
            if key in self:
                out_index = self._token2idx[key]
                if out_index in self._forced_unks:
                    out_index = self.unk_index
            elif self.use_unk:
                out_index = self.unk_index
            else:  # case: frozen, don't want unks, raise exception
                raise Exception(f"Vocabulary is frozen. Key '{key}' not found.")
        # train and token in vocab: return index and increase count
        elif key in self._token2idx:
            out_index = self._token2idx[key]
            self._counts[out_index] += 1
        # train and token not in vocab: add key to vocab, increase next index
        # and initialise count
        else:
            out_index = self._token2idx[key] = self._next_index
            self._next_index += 1
            self._idx2token[out_index] = key
            self._counts[out_index] = 1

        return out_index

    def get_index(self, index):
        try:
            return self._idx2token[index]
        except KeyError:
            raise Exception(f"Index {index} not in Vocabulary")

    def freeze(self):
        # Set frozen and include in _forced_unks all token appearing less than
        # _frequency_cutoff
        self._frozen = True
        if self._frequency_threshold:
            for token, count in reversed(self._counts.most_common()):
                if count < self._frequency_threshold:
                    self._forced_unks[token]
                else:
                    break

    def unfreeze(self):
        self._frozen = False

    #
    # FROM HERE, NOT REALLY USEFUL METHODS

    def map(self, sequence):
        if self.use_start_end:
            yield self.start_index

        for item in sequence:
            yield self[item]

        if self.use_start_end:
            yield self.end_index

    def iterkeys(self):
        for k in self._token2idx.keys():
            if k == self.unk_token or k == self.pad_token:
                continue
            else:
                yield k

    def keys(self):
        return list(self.iterkeys())

    def iteritems(self):
        for key, value in self._token2idx.items():
            if key in [self.unk_token, self.pad_token]:
                continue
            yield key, value

    def items(self):
        return list(self.iteritems())

    def values(self):
        return [value for _, value in self.iteritems()]

    def get_counts(self):
        return {self._idx2token[i]: count for i, count in self._counts.items()}

    def get_count(self, token=None, index=None):
        """Get token by token OR index"""
        if token is None and index is None:
            return None
        elif token and index is not None:
            print("Cannot do two things at once; choose one")
        elif token:
            return self._counts[self[token]]
        elif index is not None:
            return self._counts[index]
        else:
            raise Exception("impossible condition")

    @property
    def unk_index(self):
        if self.unk_token not in self:
            return None
        return self._token2idx[self.unk_token]

    @property
    def pad_index(self):
        if self.pad_token not in self:
            return None
        return self._token2idx[self.pad_token]

    @property
    def start_index(self):
        if self.start_token not in self:
            return None
        return self._token2idx[self.start_token]

    @property
    def end_index(self):
        if self.end_token not in self:
            return None
        return self._token2idx[self.end_token]

    def __repr__(self):
        return f"<Vocabulary(size={len(self)},frozen={ self._frozen})>"

    def get_serializable_contents(self):
        """
        Creats a dict containing the necessary information to recreate this instance
        """
        config = {"_token2idx": self._token2idx,
                  "_idx2token": self._idx2token,
                  "_frozen": self._frozen,
                  "_next_index": self._next_index,
                  "_counts": list(self._counts.items()),
                  "_frequency_threshold": self._frequency_threshold,
                  "use_unk": self.use_unk,
                  "unk_token": self.unk_token,
                  "use_pad": self.use_pad,
                  "pad_token": self.pad_token,
                  "use_start_end": self.use_start_end,
                  "start_token": self.start_token,
                  "end_token": self.end_token}
        return config

    @classmethod
    def deserialize_from_contents(cls, content):
        """
        Recreate a Vocabulary instance; expects same dict as output in
        `get_serializable_contents`
        """
        try:
            _mapping = content.pop("_token2idx")
            _flip = content.pop("_idx2token")
            _i = content.pop("_next_index")
            _frozen = content.pop("_frozen")
            _counts = content.pop("_counts")
            _frequency_threshold = content.pop("_frequency_threshold")
        except KeyError:
            raise Exception("unable to deserialize vocabulary")
        if isinstance(list(_flip.keys())[0], six.string_types):
            _flip = {int(k): v for k, v in _flip.items()}
        out = cls(**content)
        out._token2idx = _mapping
        out._idx2token = _flip
        out._next_index = _i
        out._counts = Counter(dict(_counts))
        out._frequency_threshold = _frequency_threshold

        if _frozen:
            out.freeze(out.use_unk)

        return out
