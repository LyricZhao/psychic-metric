import numpy
import random
from sklearn import metrics


from homework import *


if __name__ == '__main__':
    # Binary
    pred = np.array([random.random() for i in range(100)])
    target = np.array([random.randint(0, 1) for i in range(100)])
    print(binary_classification_metrics(pred, target))

    # Multi-class classification
    n = 200
    pred = np.array([random.randint(0, 9) for i in range(n)])
    target = np.array([random.randint(0, 9) for i in range(n)])
    print(multiclass_classification_metrics(pred, target))

    # Multi-label classification
    # n = 300
    # pred = np.array([[random.random() for j in range(4)] for i in range(n)])
    # target = np.array([[random.randint(0, 1) for j in range(4)] for i in range(n)])
    # print(multilabel_classification_metrics(pred, target))

    # Ranking metrics
    n = 30
    pred = [random.random() for i in range(n)]
    rel = [random.random() for i in range(n)]
    print(ranking_metrics(pred, rel))
