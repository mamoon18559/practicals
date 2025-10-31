https://www.kaggle.com/datasets/sonialikhan/heart-attack-analysis-and-prediction-dataset?utm_source=chatgpt.com

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("heart.csv")

# Drop unnecessary columns
df = df.drop(['oldpeak', 'slp', 'thall'], axis=1,errors="ignore")

# Initial exploration
print(df.head())
print(df.shape)
print(df.isnull().sum())

df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# Age distribution
plt.figure(figsize=(20, 10))
plt.title("Age of Patients")
sns.countplot(x='age', data=df)
plt.show()

# Sex distribution
plt.figure(figsize=(20, 10))
plt.title("Sex of Patients (0=Female, 1=Male)")
sns.countplot(x='sex', data=df)
plt.show()

# Chest pain type distribution
cp_data = df['cp'].value_counts().reset_index()
cp_data.columns = ['cp', 'count']
cp_data['cp'] = cp_data['cp'].astype(str)
cp_data.loc[3, 'cp'] = 'asymptomatic'
cp_data.loc[2, 'cp'] = 'non-anginal'
cp_data.loc[1, 'cp'] = 'Atypical Angina'
cp_data.loc[0, 'cp'] = 'Typical Angina'

plt.figure(figsize=(20, 10))
plt.title("Chest Pain of Patients")
sns.barplot(x=cp_data['cp'], y=cp_data['count'])
plt.show()

# ECG data distribution
ecg_data = df['restecg'].value_counts().reset_index()
ecg_data.columns = ['index', 'restecg']
ecg_data.loc[0, 'index'] = 'normal'
ecg_data.loc[1, 'index'] = 'having ST-T wave abnormality'
ecg_data.loc[2, 'index'] = 'showing probable or definite LV hypertrophy'

plt.figure(figsize=(20, 10))
plt.title("ECG Data of Patients")
sns.barplot(x=ecg_data['index'], y=ecg_data['restecg'])
plt.show()

#sns.pairplot(df,hue='output',data=df)

# Distribution plots
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.histplot(df['trtbps'], kde=True, color='magenta')
plt.xlabel("Resting Blood Pressure (mmHg)")

plt.subplot(1, 2, 2)
sns.histplot(df['thalachh'], kde=True, color='teal')
plt.xlabel("Maximum Heart Rate Achieved (bpm)")
plt.show()

plt.figure(figsize=(10, 10))
sns.histplot(df['chol'], kde=True, color='red')
plt.xlabel("Cholesterol")
plt.show()

# Standardize data
scale = StandardScaler()
df_scaled = scale.fit_transform(df)
df = pd.DataFrame(df_scaled, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'caa', 'output'])

# Train-test split
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Label encoding
lbl = LabelEncoder()
encoded_y = lbl.fit_transform(y_train)

### Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, encoded_y)
encoded_ytest = lbl.transform(y_test)
Y_pred1 = logreg.predict(X_test)
lr_conf_matrix = confusion_matrix(encoded_ytest, Y_pred1)
lr_acc_score = accuracy_score(encoded_ytest, Y_pred1)
print(f"Logistic Regression Accuracy: {lr_acc_score*100:.2f}%")

### Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, encoded_y)
ypred2 = tree.predict(X_test)
tree_conf_matrix = confusion_matrix(encoded_ytest, ypred2)
tree_acc_score = accuracy_score(encoded_ytest, ypred2)
print(f"Decision Tree Accuracy: {tree_acc_score*100:.2f}%")

### Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, encoded_y)
ypred3 = rf.predict(X_test)
rf_conf_matrix = confusion_matrix(encoded_ytest, ypred3)
rf_acc_score = accuracy_score(encoded_ytest, ypred3)
print(f"Random Forest Accuracy: {rf_acc_score*100:.2f}%")

### K Nearest Neighbour
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, encoded_y)  # fixed variable name
    pred = knn.predict(X_test)   # fixed variable name
    error_rate.append(np.mean(pred != encoded_ytest))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.title('Error Rate vs. K Value')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.title('Error Rate vs. K Value')
plt.show()

### Support Vector Machine
svm_model = svm.SVC()
svm_model.fit(X_train, encoded_y)
ypred5 = svm_model.predict(X_test)
svm_conf_matrix = confusion_matrix(encoded_ytest, ypred5)
svm_acc_score = accuracy_score(encoded_ytest, ypred5)
print(f"SVM Accuracy: {svm_acc_score*100:.2f}%")

# Select best K
best_k = error_rate.index(min(error_rate)) + 1
print(f"Best K value: {best_k}")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, encoded_y)  # fixed variable name
ypred4 = knn.predict(X_test) # fixed variable name

knn_conf_matrix = confusion_matrix(encoded_ytest, ypred4)
knn_acc_score = accuracy_score(encoded_ytest, ypred4)  # now properly defined
print(f"KNN Accuracy (k={best_k}): {knn_acc_score*100:.2f}%")

model_acc = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'K Nearest Neighbor', 'SVM'],
    'Accuracy': [lr_acc_score*100, tree_acc_score*100, rf_acc_score*100, knn_acc_score*100, svm_acc_score*100]
})
model_acc = model_acc.sort_values(by=['Accuracy'], ascending=False)
print(model_acc)