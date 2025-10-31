# dataset link: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
from sklearn.model_selection import train_test_split

sns.set_theme(style="darkgrid")

data = pd.read_csv("data.csv")

print(data)
print(data.info())
print(data.describe().T)

data.drop(["id"],inplace=True,errors="ignore")
data["diagnosis"] = data["diagnosis"].map({"B":0,"M":1})
print(data)

# One hot And Imputing

data = pd.get_dummies(data,drop_first=True)
print(data)

features = data.drop(["diagnosis"],axis=1)
target = data["diagnosis"]

ss = SimpleImputer(strategy="mean")

imputed_features = ss.fit_transform(features.values)
imputed_features = pd.DataFrame(imputed_features,columns=features.columns)

plt.figure(figsize=(12,10))
corr = imputed_features.corr()
sns.heatmap(corr,cmap="coolwarm",annot=False)
plt.title("Feature Corr")
plt.show()

plt.figure(figsize=(15,8))
sns.boxplot(data=imputed_features,fliersize=2)
plt.xticks(rotation=90)
plt.show()

x_train,x_test,y_train,y_test = train_test_split(imputed_features,target,test_size=0.2,stratify=target)

ss = StandardScaler()
scaled_x_train = ss.fit_transform(x_train)
scaled_x_test = ss.transform(x_test)

model = LogisticRegression()
model.fit(scaled_x_train,y_train)

cr = classification_report(y_test,model.predict(scaled_x_test))
print(cr)

y_pred = model.predict(scaled_x_test)
results = {
    'Accuracy' : accuracy_score(y_test,y_pred),
    'Precision':precision_score(y_test,y_pred),
    'Recall':recall_score(y_test,y_pred),
    'F1 Score':f1_score(y_test,y_pred)
}
print(results)

cm = confusion_matrix(y_test,y_pred)
print(cm)

summary = pd.DataFrame([['Logistic Regression']+list(results.values())],columns=['Model']+list(results.keys()))
print(summary)