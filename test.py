import random
import numpy as np

from sklearn import metrics
from homework import *


if __name__ == '__main__':
    # Binary
    pred = np.array([random.random() for i in range(100)])
    target = np.array([random.randint(0, 1) for i in range(100)])
    print(metrics.accuracy_score(target, pred >= 0.5))
    print(metrics.precision_score(target, pred >= 0.5))
    print(metrics.recall_score(target, pred >= 0.5))
    print(metrics.f1_score(target, pred >= 0.5))
    print(metrics.roc_auc_score(target, pred))
    print(binary_classification_metrics(pred, target))
    print()

    # Multi-class classification
    n = 200
    pred = np.array([random.randint(0, 9) for i in range(n)])
    target = np.array([random.randint(0, 9) for i in range(n)])
    print(metrics.accuracy_score(target, pred))
    print(metrics.precision_score(target, pred, average='macro'))
    print(metrics.recall_score(target, pred, average='macro'))
    print(metrics.f1_score(target, pred, average='macro'))
    print(multiclass_classification_metrics(pred, target))
    print()

    # Multi-label classification
    n = 300
    pred = np.array([[random.random() for j in range(4)] for i in range(n)])
    target = np.array([[random.randint(0, 1) for j in range(4)] for i in range(n)])
    print(metrics.accuracy_score(target, pred >= 0.5))
    print(metrics.precision_score(target, pred >= 0.5, average='macro'))
    print(metrics.recall_score(target, pred >= 0.5, average='macro'))
    print(metrics.f1_score(target, pred >= 0.5, average='macro'))
    print(metrics.precision_score(target, pred >= 0.5, average='micro'))
    print(metrics.recall_score(target, pred >= 0.5, average='micro'))
    print(metrics.f1_score(target, pred >= 0.5, average='micro'))
    print(multilabel_classification_metrics(pred, target))
    print()

    # Ranking metrics
    n = 30
    pred = [random.random() for i in range(n)]
    rel = [random.random() for i in range(n)]
    print(ranking_metrics(pred, rel))
