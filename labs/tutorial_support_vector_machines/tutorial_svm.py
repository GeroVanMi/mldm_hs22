import pandas as pd
from matplotlib import pyplot as plt
from seaborn import load_dataset, pairplot
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

df = load_dataset('penguins')
# 11 rows have N/A values. We drop them for this tutorial.
df = df.dropna()
pairplot(df, hue='species')
plt.show()

X = df.drop(columns=['species'])
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)

column_transformer = make_column_transformer(
    (OneHotEncoder(), ['sex', 'island']),
    (StandardScaler(), ['bill_depth_mm', 'bill_length_mm', 'flipper_length_mm', 'body_mass_g']),
    remainder='passthrough'
)

# Transform the X columns
X_train = column_transformer.fit_transform(X_train)
X_train = pd.DataFrame(data=X_train, columns=column_transformer.get_feature_names_out())

X_test = column_transformer.fit_transform(X_test)
X_test = pd.DataFrame(data=X_test, columns=column_transformer.get_feature_names_out())

# Run a Grid Search over the following parameters. 5x5x1 = 25
params = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, .1, .01, .001, .00001],
    'kernel': ['rbf']
}

classifier = GridSearchCV(
    estimator=SVC(),
    param_grid=params,
    cv=5,  # Fitting 5 folds for each of the 25 parameter combinations = 125 folds
    n_jobs=5,
    verbose=1  # Configure console debug output
)

classifier.fit(X_train, y_train)
print(classifier.best_params_)  # C: 1000, gamma: 0.01, kernel: 'rbf'

model = SVC(kernel='rbf', gamma=0.01, C=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Passing in our own data:
penny = [{
    'sex': 'Female',
    'island': 'Torgersen',
    'bill_depth_mm': 23.3,
    'bill_length_mm': 43.5,
    'flipper_length_mm': 190,
    'body_mass_g': 4123
}]
penny = pd.DataFrame(penny)
df_transformed = column_transformer.transform(penny)
df_transformed = pd.DataFrame(df_transformed, columns=column_transformer.get_feature_names_out())
predicted_species = model.predict(df_transformed)
print(predicted_species)  # Returns: ['Adelie']
