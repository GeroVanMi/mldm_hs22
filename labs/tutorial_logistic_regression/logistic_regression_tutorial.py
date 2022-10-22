"""
author: Gerome Meyer
Based on this tutorial: https://realpython.com/logistic-regression-python/#logistic-regression-in-python
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Create an input array from 0-9 (10 values) and reshape it to be in a column.
# This is the so-called predictor.
x = np.arange(10).reshape(-1, 1)
# [[0]
#  [1]
#  ...
#  [8]
#  [9]]

# y is the value we want to predict. In this case we have two classes: 0 and 1. (Since we're using logistic regression.)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Other solvers available are: 'newton-cg', 'lbfgs', 'sag', 'saga', ('liblinear' = default)

# We need to match the solver and regularization method carefully:
# 'liblinear' solver doesn’t work without regularization.
# 'newton-cg', 'sag', 'saga', and 'lbfgs' don’t support L1 regularization.
# 'saga' is the only solver that supports elastic-net regularization.
model = LogisticRegression(solver='liblinear', random_state=0)

# Now we fit the model
model.fit(x, y)

# This shows us how many classes we have. (In our case only 0 and 1 => 2 classes)
print(model.classes_)

# Make a prediction and show the probabilities
print(model.predict_proba(x))

plt.scatter(x, y)
plt.plot(model.predict(x))
plt.show()

# Get the model score for a certain prediction
# This makes predictions for x and then matches them against the actual values y
print(f"Model Score: {model.score(x, y)}")

# This gives us the confusion matrix for our prediction.
# [[TN, FP], (True Negatives, False Positives)
#   FN, TP]] (False Negatives, True Positives)
cm = confusion_matrix(y, model.predict(x))
print(cm)

# Code to visualize the confusion matrix:
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red', size=28)
plt.show()

print(classification_report(y, model.predict(x)))

# We can now tweak our model to try and improve these stats
# We set the regularization strength from 1.0 to 10.0
model = LogisticRegression(solver='liblinear', random_state=0, C=10)
model.fit(x, y)

# Now we can see that it predicts the value at index 3 correctly.
plt.plot(x, model.predict(x))
plt.scatter(x, y)
plt.show()

# We can see this by looking at the score as well:
print(f"Tweaked model score: {model.score(x, y)}")
# Same goes for the classification report.
print(classification_report(y, model.predict(x)))
