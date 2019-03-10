import logging

import torch
import numpy as np
from sklearn_crfsuite.metrics import flat_f1_score, flat_classification_report

from data_preparation import batches_generator

logger = logging.getLogger(__name__)


def predict_tags(model, batch_tokens, batch_lengths, idx2tag, batch_tags=None):
    """Performs predictions and transforms indices to tokens and tags."""

    tag_scores = model(batch_tokens, batch_lengths)
    predicted_tags = np.vectorize(idx2tag.get)(
        torch.argmax(tag_scores, dim=2).data.numpy())
    if batch_tags is not None:
        return predicted_tags, model.loss(tag_scores, batch_tags)
    return predicted_tags


def my_scorer(true_tags, predicted_tags, sorted_labels):
    score = flat_f1_score(
        true_tags, predicted_tags, average='weighted', labels=sorted_labels
    )
    logger.info(f"f1 score: {score:.3f}")
    print(flat_classification_report(
        true_tags, predicted_tags, labels=sorted_labels, digits=3
    ))


def eval_model_for_set(model, tokens, tags, token2idx, tag2idx, sorted_labels):
    """Computes NER quality measures given model and a dataset"""
    model.eval()
    predicted_tags, true_tags, loss = [], [], 0
    idx2tag = dict((v, k) for (k, v) in tag2idx.items())
    with torch.no_grad():
        loader = batches_generator(
            len(tokens), tokens, tags, token2idx, tag2idx
        )
        for x_batch, y_batch, lengths in loader:
            padded_pred_tags, batch_loss = predict_tags(
                model, x_batch, lengths, idx2tag, y_batch,
            )
            loss += batch_loss
            padded_true_tags = np.vectorize(idx2tag.get)(y_batch.data)
            for x, y, l in zip(padded_pred_tags, padded_true_tags, lengths):
                predicted_tags.append(x[:l])
                true_tags.append(y[:l])
        my_scorer(true_tags, predicted_tags, sorted_labels)
        return loss / len(true_tags)
