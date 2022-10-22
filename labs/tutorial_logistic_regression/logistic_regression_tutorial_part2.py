"""
author: Gerome Meyer
Based on the following article:
https://realpython.com/logistic-regression-python
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Create data
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1])

# Create model
model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(x, y)

# Evaluate Model
probabilities = model.predict_proba(x)
y_predictions = model.predict(x)
model_score = model.score(x, y)
conf_matrix = confusion_matrix(y, y_predictions)
report = classification_report(y, y_predictions)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(conf_matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red', size=28)
plt.show()

plt.scatter(x, y)
plt.plot(model.predict(x))
plt.show()