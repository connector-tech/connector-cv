import torch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def calculate_metrics(outputs, labels):
    """
    Calculates accuracy, precision, recall, and F1-score.

    Args:
        outputs (torch.Tensor): Model outputs.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        tuple: Accuracy, precision, recall, F1-score.
    """
    with torch.no_grad():
        predictions = torch.round(torch.sigmoid(outputs))
        correct = (predictions == labels).sum().float()
        accuracy = correct / len(labels)

        tp = (predictions & labels).sum().float()
        tn = ((~predictions) & (~labels)).sum().float()
        fp = (predictions & (~labels)).sum().float()
        fn = ((~predictions) & labels).sum().float()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return accuracy, precision, recall, f1_score


def calculate_roc_auc(outputs, labels):
    """
    Calculates the Receiver Operating Characteristic (ROC) AUC score.

    Args:
        outputs (torch.Tensor): Model outputs.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        float: ROC AUC score.
    """
    with torch.no_grad():
        predictions = torch.sigmoid(outputs)
        return roc_auc_score(labels.cpu().numpy(), predictions.cpu().numpy())


def calculate_average_precision(outputs, labels):
    """
    Calculates the average precision score.

    Args:
        outputs (torch.Tensor): Model outputs.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        float: Average precision score.
    """
    with torch.no_grad():
        predictions = torch.sigmoid(outputs)
        return average_precision_score(labels.cpu().numpy(), predictions.cpu().numpy())


def calculate_confusion_matrix(outputs, labels):
    """
    Calculates the confusion matrix.

    Args:
        outputs (torch.Tensor): Model outputs.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        torch.Tensor: Confusion matrix.
    """
    with torch.no_grad():
        predictions = torch.round(torch.sigmoid(outputs))
        tn = ((~predictions) & (~labels)).sum().int()
        fp = (predictions & (~labels)).sum().int()
        fn = ((~predictions) & labels).sum().int()
        tp = (predictions & labels).sum().int()
        return torch.tensor([[tn, fp], [fn, tp]])
