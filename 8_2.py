#https://www.kaggle.com/datasets/prasad22/healthcare-dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
import lime
from lime import lime_tabular

import warnings
warnings.filterwarnings('ignore')

"""##2. Load Dataset"""

df = pd.read_csv('healthcare_dataset.csv')
print("Dataset shape:", df.shape)
df.head()

"""##3. Data Preprocessing"""

print(df.isnull().sum())
df = df.dropna()

cat_cols = df.select_dtypes(include=['object']).columns.tolist()

cols_to_exclude = ['Name', 'Doctor', 'Hospital', 'Insurance Provider', 'Date of Admission', 'Discharge Date']
cat_cols_to_encode = [col for col in cat_cols if col not in cols_to_exclude]

df['Medical Condition'] = df['Medical Condition'].str.title()
df['Gender'] = df['Gender'].str.title()
df['Test Results'] = df['Test Results'].str.title()

df_encoded = pd.get_dummies(df, columns=[col for col in cat_cols_to_encode if col != 'Test Results'], drop_first=True)

print("\nColumns after One-Hot Encoding:")
print(df_encoded.columns.tolist())

TARGET_COLUMN = 'Test Results'

le = LabelEncoder()
df_encoded[TARGET_COLUMN] = le.fit_transform(df_encoded[TARGET_COLUMN])
target_names = le.classes_

X = df_encoded.drop(columns=cols_to_exclude + [col for col in df.columns if col not in df_encoded.columns] + [TARGET_COLUMN], errors='ignore')
y = df_encoded[TARGET_COLUMN]

feature_names = X.columns.tolist()

"""## 4. Data Splitting and Scaling"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

"""## 5. Model Training"""

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled_df, y_train)

"""## 6. Model Evaluation"""

y_pred = model.predict(X_test_scaled_df)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_names)
cm = confusion_matrix(y_test, y_pred)
print("\n--- Model Performance Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)

"""##7. Explainable AI (XAI) Implementation"""

feature_importances = model.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]
top_3_features = [feature_names[i] for i in sorted_idx[:3]]
print(f"\nTop 3 Global Features (RF Importance): {top_3_features}")

"""###1. Global Feature Importance (Mean Absolute SHAP values across all classes)"""

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances[sorted_idx][:10], y=np.array(feature_names)[sorted_idx][:10])
plt.title('Random Forest Feature Importance (Top 10)')
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.show()

"""###2. SHAP Dependence Plot for a key feature (e.g., 'Age')"""

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

instance_index = 42
true_label = target_names[y_test.iloc[instance_index]]
predicted_label = target_names[model.predict(X_test_scaled_df.iloc[[instance_index]])[0]]
print(f"\nConceptual Local Explanation (Instance {instance_index}): True Label: {true_label}, Predicted Label: {predicted_label}")

explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train_scaled_df),
    feature_names=feature_names,
    class_names=target_names,
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

explanation = explainer_lime.explain_instance(
    data_row=X_test_scaled_df.iloc[instance_index].values,
    predict_fn=model.predict_proba,
    num_features=10
)

print("\nLIME Local Explanation Plot (for instance {}):".format(instance_index))
explanation.show_in_notebook(show_table=True)