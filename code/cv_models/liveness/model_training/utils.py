import torch

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, roc_auc_score


def calculate_metrics(outputs, labels):
    """
    Calculates accuracy, precision, recall, and F1-score for binary classification.

    Args:
        outputs (torch.Tensor): Model outputs (logits or probabilities).
        labels (torch.Tensor): Ground truth labels (0 or 1).

    Returns:
        tuple: (accuracy, precision, recall, f1_score)
    """

    threshold = 0.5
    predictions = (outputs > threshold).float()

    accuracy = (predictions == labels).float().mean()

    num_positives = torch.sum(labels)
    num_predicted_positives = torch.sum(predictions)
    if num_positives > 0 and num_predicted_positives > 0:
        precision = torch.sum(predictions * labels).float() / num_predicted_positives
        recall = torch.sum(predictions * labels).float() / num_positives
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        precision = torch.tensor(0.0)
        recall = torch.tensor(0.0)
        f1_score = torch.tensor(0.0)

    return accuracy, precision, recall, f1_score


def calculate_confusion_matrix(outputs, labels):
    """
    Calculates the confusion matrix for binary classification.

    Args:
        outputs (torch.Tensor): Model outputs (logits or probabilities).
        labels (torch.Tensor): Ground truth labels (0 or 1).

    Returns:
        torch.Tensor: Confusion matrix (2x2) with integer counts.
    """
    labels = labels.to(outputs.device)
    threshold = 0.5
    predictions = (outputs > threshold).int()

    if not labels.dtype in (torch.long, torch.int):
        labels = labels.long()

    cm = torch.zeros((2, 2), dtype=torch.int64, device=labels.device)

    for i in range(len(labels)):
        cm[labels[i], predictions[i]] += 1

    return cm


def calculate_roc_auc(outputs, labels):
    """
    Calculates the Receiver Operating Characteristic (ROC) Area Under Curve (AUC).

    Args:
        outputs (torch.Tensor): Model outputs (logits).
        labels (torch.Tensor): Ground truth labels.

    Returns:
        float: ROC AUC score.
    """
    outputs = outputs.cpu().detach()
    labels = labels.cpu().detach()

    labels = labels.long()

    roc_auc = roc_auc_score(labels, outputs)

    return roc_auc


def calculate_average_precision(outputs, labels):
    """
    Calculates the Average Precision (AP) using the scikit-learn implementation.

    Args:
        outputs (torch.Tensor): Model outputs (logits).
        labels (torch.Tensor): Ground truth labels.

    Returns:
        float: Average Precision score.
    """
    outputs = outputs.cpu().detach()
    labels = labels.cpu().detach()

    labels = labels.long()

    average_precision = average_precision_score(labels, outputs)

    return average_precision
