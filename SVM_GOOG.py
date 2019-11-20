

# FI6151 SVM Group Assignment
# GROUP : 7
# STOCK : GOOGLE (GOOG)
MEMBERS : QIANG MA (19054386) & AKHIL MENON (19008414)



### Import Data
"""

# Commented out IPython magic to ensure Python compatibility.
# Import machine learning libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from pandas_datareader import data
from sklearn import svm, preprocessing, metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

# Config to only draw static images and store in the notebook 
# %matplotlib inline

"""### Prepare Data"""

# Mount google drive as local drive. Authorization code is needed on first time execution.
from google.colab import drive
drive.mount('/content/gdrive');

# Read data from excel, i.e. date and four features.
tmp_df = pd.read_excel (r'/content/gdrive/My Drive/Computational Finance Course/Machine Learning/3YGOOG.xlsx', header = 0, usecols = "A:E");

# Assign meaningful column names.
tmp_df.columns = ["Dates", "US10YBOND", "NIKKEI", "VOLUME", "GOOG_PCT"];
print(tmp_df.head(2));

# Shit GOOG_PCT to create target feature.
tmp_df["TARGET_PCT"] = tmp_df["GOOG_PCT"].shift(1);

# Convert all features to percent change to unify scale. GOOG_PCT is already in percent change format from raw data.
tmp_df["US10YBOND_PCT"] = tmp_df["US10YBOND"].pct_change(3);
tmp_df["NIKKEI_PCT"] = tmp_df["NIKKEI"].pct_change(3);
tmp_df["VOLUME_PCT"] = tmp_df["VOLUME"].pct_change(1);

# Drop N/A value
tmp_df = tmp_df.dropna();

# Construct training features and target feature
X = pd.concat([tmp_df["US10YBOND_PCT"], tmp_df["NIKKEI_PCT"], tmp_df["VOLUME_PCT"], tmp_df["GOOG_PCT"]], axis=1);

# Construct target feature, change positive change to 1 and negtive to 0
y = pd.DataFrame(tmp_df["TARGET_PCT"]);
y[ y > 0 ] = 1;
y[ y < 0 ] = 0;

print(pd.concat([X, y], axis=1).head(2));

"""### Visualize Data
?? We have a problem that the plot does't reflect the cor table.
"""

# Relation between US Bond and GOOG in percent change
plt.scatter(X["US10YBOND_PCT"], tmp_df["TARGET_PCT"]);
plt.title("Relation between US Bond and GOOG in percent change");
plt.ylabel("US 10 Years bond percent change");
plt.xlabel("Google percent change");
plt.show();

# Relation between NIKKEI Bond and GOOG in percent change
plt.scatter(X["NIKKEI_PCT"], tmp_df["TARGET_PCT"]);
plt.title("Relation between NIKKEI and GOOG in percent change");
plt.ylabel("NIKKEI percent change");
plt.xlabel("Google percent change");
plt.show();

# Relation between VOLUME Bond and GOOG in percent change
plt.scatter(X["VOLUME_PCT"], tmp_df["TARGET_PCT"]);
plt.title("Relation between VOLUME and GOOG in percent change");
plt.ylabel("VOLUME percent change");
plt.xlabel("Google percent change");
plt.show();

# Correlation bewteen four features
pd.concat([X, tmp_df["TARGET_PCT"]], axis=1).corr().style.background_gradient(cmap='coolwarm')

"""### Split data"""

# Split data into 0.8 training set and 0.2 testing set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, shuffle = False);
print(X_train.head(2));
print(X_test.head(2));

"""### Train Model"""

# Define parameters for SVM
Kernels = ['linear', 'rbf', 'sigmoid'];
Gammas = [1e-2, 1e-3, 1e-4];
Cs = [1, 10, 100, 1000, 10000];
i = 0;

# Iterate over all combination, to find the parameters that fot best.
for k in Kernels :
  for g in Gammas :
    for c in Cs :
      # Construct SVM model
      clf = svm.SVC(kernel=k, gamma=g, C=c);

      # Train model on training set
      clf.fit(X, y["TARGET_PCT"]);

      # Predict on testing set
      y_pred = clf.predict(X_test);

      # Print parameters
      print("Iteration", i, ": ", end = ' ');
      print("Kernel=", k, end = ' ');
      print("Gammas=", g, end = ' ');
      print("Cs=", c, end = ' ');
      print("Accuracy=", round(metrics.accuracy_score(y_test, y_pred), 4), end = ' ');
      print("Precision=", round(metrics.precision_score(y_test, y_pred), 4), end = ' ');
      print("Recall=", round(metrics.recall_score(y_test, y_pred), 4) );
      i = i+1;

"""### Best fit"""

# From the result, we found Kernel= rbf Gammas= 0.01 Cs= 10000 gives best result: Accuracy= 0.6291 Precision= 0.6186 Recall= 0.7595
# However, C = 10000 is a boundary in our trying array, so we run one more test with C=100000.
# Accuracy and Recall decreased. 
clf = svm.SVC(kernel='rbf', gamma=0.01, C=100000);
clf.fit(X, y["TARGET_PCT"]);
y_pred = clf.predict(X_test);
print("One More Test: Accuracy=", round(metrics.accuracy_score(y_test, y_pred), 4), end = ' ');
print("Precision=", round(metrics.precision_score(y_test, y_pred), 4), end = ' ');
print("Recall=", round(metrics.recall_score(y_test, y_pred), 4) );

# Finally, We choose Kernel= rbf Gammas= 0.01 Cs= 10000 gives best result: Accuracy= 0.6291 Precision= 0.6186 Recall= 0.7595
clf = svm.SVC(kernel='rbf', gamma=0.01, C=10000);
clf.fit(X, y["TARGET_PCT"]);
y_pred = clf.predict(X_test);
print("Best fit: Accuracy=", round(metrics.accuracy_score(y_test, y_pred), 4), end = ' ');
print("Precision=", round(metrics.precision_score(y_test, y_pred), 4), end = ' ');
print("Recall=", round(metrics.recall_score(y_test, y_pred), 4) );

# Get confusion matrix
cm = confusion_matrix(y_test, y_pred);

# Display confusion matrix
df_cm = pd.DataFrame(cm, range(2), range(2))
plt.figure(figsize = (3,2))
sn.set(font_scale=2.0)
sn.heatmap(df_cm, annot=True, fmt="d");
plt.show();

"""### Simulate Trade"""

# Get prediction data again as it was converted 1/0 in previous step.
tmp_x_train, tmp_x_test, tmp_y_train, tmp_y_test = train_test_split(X, tmp_df["TARGET_PCT"], test_size = 0.20, shuffle = False);

# Roughly calculate market return by sum the percent change over the period.
Stock_ret = np.sum(tmp_y_test);

g = np.zeros(tmp_y_test.shape[0]+1);
g[0] = 1;
for i in range(0, tmp_y_test.shape[0]):
  g[i+1] = g[i] * (1 + tmp_y_test.iloc[:].values[i]/100);

p = np.zeros(tmp_y_test.shape[0]+1);
p[0] = 1;
for i in range(0, tmp_y_test.shape[0]):
  p[i+1] = p[i] * (1 + y_pred[i]*tmp_y_test.iloc[:].values[i]/100);

# print(g)
# x, = plt.plot(g);
# plt.plot(p, label="SVM");
# plt.plot(g, p);
# plt.show();

x = np.linspace(0, 1, tmp_y_test.shape[0]+1);
fig, ax = plt.subplots()
ax.plot(x, g, '-b', label='Google')
ax.plot(x, p, '--r', label='SVM')
ax.axis('equal')
leg = ax.legend();

# Calculate trade return by dot product trading signal and percent change.
# Because a zero trading signal means we clear our position.
Trade_ret = np.dot(tmp_y_test, y_pred);

# print("Over the testing period, Google stock return is ", round(Stock_ret, 2), "% return.");
print("With our trading strategy, we achieve ", round(Trade_ret, 2), "% return.");

print("Market return is ", round((g[150]-1)*100, 2), "% return.");
print("With our trading strategy, we achieve ", round((p[150]-1)*100, 2), "% return.");
