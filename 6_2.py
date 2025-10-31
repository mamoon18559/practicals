# 🚀 Parkinson’s Disease Prediction — Single Cell Colab Version (UCI Dataset)
# Source: https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data

# ========== ENVIRONMENT SETUP ==========
!pip install -q numpy==1.26.4 pandas==2.2.2 scikit-learn matplotlib seaborn shap==0.44.1 lime==0.2.0.1

# ========== IMPORTS ==========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ========== LOAD DATA ==========
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df = pd.read_csv(url)

print("✅ Dataset Loaded Successfully!")
print(df.head())

# ========== DATA PREPROCESSING ==========
print("\n🔍 Checking for Missing Values:")
print(df.isnull().sum())

X = df.drop(columns=['status', 'name'])
y = df['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ========== MODEL TRAINING ==========
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ========== EVALUATION ==========
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {acc:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('🧩 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ========== FEATURE IMPORTANCE ==========
importances = pd.Series(model.feature_importances_, index=df.columns[1:-1])
importances.sort_values(ascending=False).head(10).plot(kind='bar', figsize=(8,4))
plt.title("🔥 Top 10 Feature Importances (RandomForest)")
plt.show()

# ========== EXPLAINABLE AI (SHAP) ==========
print("\n💡 Generating SHAP Explanations...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test, feature_names=df.columns[1:-1])

# ========== LOCAL INTERPRETATION (LIME) ==========
print("\n💬 Generating LIME Explanation for One Sample...")
lime_explainer = lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=df.columns[1:-1],
    class_names=['Healthy','Parkinson'],
    discretize_continuous=True
)

i = np.random.randint(0, X_test.shape[0])
exp = lime_explainer.explain_instance(X_test[i], model.predict_proba, num_features=10)
exp.show_in_notebook(show_table=True)

print("\n✅ Experiment Completed Successfully!")
