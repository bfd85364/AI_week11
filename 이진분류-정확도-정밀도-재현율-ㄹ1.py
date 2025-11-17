#예제 31-01
#http://torchmetrics.readthedocs.io/en/stable/page/classification.html
import torch
from torchmetrics import ConfusionMatrix, Accuracy, Precision, Recall, F1Score
from torchmetrics.functional import confusion_matrix, stat_scores
from torchmetrics.functional import accuracy, precision, recall, f1_score
import seaborn as sn
import matplotlib.pyplot as plt

y_true = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1, 1])
y_pred = torch.tensor([0, 1, 0, 1, 0, 0, 0, 0, 1, 0])

C = ConfusionMatrix(num_classes = 2, task='multiclass') (y_pred, y_true)
print('C=', C)

plt.figure(figsize = (6,4))
ax= sn.heatmap(C, annot = True, fmt ='d')
plt.show()

acc = accuracy(y_pred, y_true, task='multiclass')
P = precision(y_pred, y_true, task = False)
R = recall(y_pred, y_true, task = False)
f1 = f1_score(y_pred, y_true, task = False)

print('accuracy= ', acc)
print('precision= ', P)
print('recall= ', R)
print('f1_score= ', f1)

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
C = confusion_matrix(y_true, y_pred)
print('C= ', C)

acc = accuracy_score(y_true, y_pred)

p, r, f1, support= precision_recall_fscore_support
C = confusion_matrix(y_true, y_pred)
print('C= ', C)

print('accuracy= ', acc)
print('p=', p)
print('r=', r)
print('f1=',f1)
print(classification_report(y_true, y_pred))
