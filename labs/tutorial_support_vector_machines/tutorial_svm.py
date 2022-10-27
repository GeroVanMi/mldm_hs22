from seaborn import load_dataset, pairplot
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = load_dataset('penguins')
# 11 rows have N/A values. We drop them for this tutorial.
df = df.dropna()

pairplot(df, hue='species')
plt.show()

X = df[['bill_length_mm', 'bill_depth_mm']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

# SVC(
#     C=1.0,                          # The regularization parameter
#     kernel='rbf',                   # The kernel type used
#     degree=3,                       # Degree of polynomial function
#     gamma='scale',                  # The kernel coefficient
#     coef0=0.0,                      # If kernel = 'poly'/'sigmoid'
#     shrinking=True,                 # To use shrinking heuristic
#     probability=False,              # Enable probability estimates
#     tol=0.001,                      # Stopping crierion
#     cache_size=200,                 # Size of kernel cache
#     class_weight=None,              # The weight of each class
#     verbose=False,                  # Enable verbose output
#     max_iter=- 1,                   # Hard limit on iterations
#     decision_function_shape='ovr',  # One-vs-rest or one-vs-one
#     break_ties=False,               # How to handle breaking ties
#     random_state=None               # Random state of the model
# )

model = SVC(kernel='linear')