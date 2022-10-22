"""
author: Gerome Meyer
Based on the following article:
https://realpython.com/logistic-regression-python
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

x, y = load_digits(return_X_y=True)

# Plot the images
plt.figure(figsize=(10, 10))
for index in range(0, 25):
    plt.subplot(5, 5, index + 1)
    plt.title(y[index])
    img = x[index].reshape(8, 8)
    plt.imshow(img, cmap='gray')
plt.show()

# We split the data into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# The standard scaler transforms our data to values between -1 and 1.
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

model = LogisticRegression(solver='liblinear', C=1, multi_class='ovr', random_state=0)
model.fit(x_train, y_train)

# Note that we only transform (not fit_transform) for the test data.
x_test = scaler.transform(x_test)

y_prediction = model.predict(x_test)

print(f"Training score: {model.score(x_train, y_train)}")
print(f"Test score: {model.score(x_test, y_test)}")

conf_matrix = confusion_matrix(y_test, y_prediction)
# This is code to visualize a confusion matrix with matplotlib.
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(conf_matrix)
ax.grid(False)
ax.set_xlabel('Predicted outputs', fontsize=12, color='black')
ax.set_ylabel('Actual outputs', fontsize=12, color='black')
ax.xaxis.set(ticks=range(10))
ax.yaxis.set(ticks=range(10))
ax.set_ylim(9.5, -0.5)
for i in range(10):
    for j in range(10):
        ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color='white')
plt.show()

# This thing is spot-on...
plt.figure(figsize=(10, 10))
for index in range(0, 25):
    plt.subplot(5, 5, index + 1)
    plt.title(y_prediction[index])
    img = x_test[index].reshape(8, 8)
    plt.imshow(img, cmap='gray')
plt.show()
