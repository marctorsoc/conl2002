import logging

import torch
import numpy as np
from sklearn_crfsuite.metrics import flat_f1_score, flat_classification_report
from torch.utils.data import DataLoader

from data_preparation import pad_and_sort_batch

logger = logging.getLogger(__name__)


def predict_tags(model, batch_tokens, batch_lengths, batch_tags=None, using_crf=False):
    """Performs predictions and transforms indices to tokens and tags."""

    if not using_crf:
        tag_scores = model(batch_tokens, batch_lengths)
        predicted_tags = torch.argmax(tag_scores, dim=2).data.numpy()
        # for test, just return predicted_tags
        if batch_tags is None:
            return predicted_tags
        # for train/validation sets, return the loss as well
        else:
            loss = model.loss(tag_scores, batch_tags)
            return predicted_tags, loss
    else:
        tag_scores, mask = model(batch_tokens, batch_lengths)
        predicted_tags = model.decode(tag_scores, mask)
        # for test, just return predicted_tags
        if batch_tags is None:
            return predicted_tags
        # for train/validation sets, return the loss as well
        else:
            loss = model.loss(tag_scores, mask, batch_tags)
            return predicted_tags, loss


def my_scorer(true_tags, predicted_tags, sorted_labels):
    score = flat_f1_score(
        true_tags, predicted_tags, average="weighted", labels=sorted_labels
    )
    logger.info(f"f1 score: {score:.3f}")
    print(
        flat_classification_report(
            true_tags, predicted_tags, labels=sorted_labels, digits=3
        ),
        flush=True,
    )


def eval_model_for_set(model, data, vectorizer, using_crf=False):
    """Computes NER quality measures given model and a dataset"""
    # TODO: documentation
    model.eval()
    predicted_tags, true_tags, loss = [], [], 0
    with torch.no_grad():
        loader = DataLoader(
            data, batch_size=len(data), shuffle=True, collate_fn=pad_and_sort_batch
        )
        # TODO: this is extremely ugly
        for batch in loader:
            x_batch, y_batch, lengths = batch
        padded_pred_tags, loss = predict_tags(
            model, x_batch, lengths, y_batch, using_crf
        )
        try:
            padded_pred_tags = vectorizer.map_sequence_back(
                vectorizer.tag_vocab, padded_pred_tags
            )
        except:
            padded_pred_tags = [
                np.vectorize(vectorizer.tag_vocab.get_index)(doc_padded_pred_tags)
                for doc_padded_pred_tags in padded_pred_tags
            ]
        padded_true_tags = vectorizer.map_sequence_back(
            vectorizer.tag_vocab, y_batch.data
        )
        for x, y, l in zip(padded_pred_tags, padded_true_tags, lengths):
            if not using_crf:
                predicted_tags.append(x[:l])
            true_tags.append(y[:l])
        if using_crf:
            predicted_tags = padded_pred_tags
        my_scorer(true_tags, predicted_tags, sorted(vectorizer.tag_vocab.keys()))
        return loss
