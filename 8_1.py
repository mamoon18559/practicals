# 🚀 Explainable AI in Healthcare (SHAP + LIME) — One Cell (Local CSV Upload)
https://www.kaggle.com/datasets/prasad22/healthcare-dataset
# ========== ENVIRONMENT FIX ==========
!pip install -q numpy==1.26.4 pandas==2.2.2 shap==0.44.1 lime==0.2.0.1 scikit-learn matplotlib seaborn
!pip uninstall -y jax jaxlib cudf-cu12 dask-cudf-cu12 pytensor opencv-python opencv-python-headless opencv-contrib-python > /dev/null 2>&1

# ========== UPLOAD DATA ==========
from google.colab import files
import pandas as pd, numpy as np, warnings, seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
import lime
from lime import lime_tabular

warnings.filterwarnings('ignore')

print("📁 Upload your 'healthcare_dataset.csv' file")
uploaded = files.upload()

# Find uploaded file
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)
print(f"✅ Data Loaded: {filename} — Shape:", df.shape)
print("\n🔍 Columns:", df.columns.tolist())

# ========== DATA CLEANING ==========
cols_to_exclude = ['Name', 'Doctor', 'Hospital', 'Insurance Provider', 'Date of Admission', 'Discharge Date']
df = df.drop(columns=cols_to_exclude, errors='ignore')

# Standardize capitalization
for col in ['Medical Condition', 'Gender', 'Test Results']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.title()

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# ========== SPLIT DATA ==========
X = df.drop('Test Results', axis=1)
y = df['Test Results']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ========== TRAIN MODEL ==========
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n🎯 Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ========== CONFUSION MATRIX ==========
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# ========== SHAP EXPLAINABILITY ==========
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

plt.title("SHAP Summary Plot")
shap.summary_plot(shap_values, X_train, feature_names=X.columns)

# ========== LIME EXPLAINABILITY ==========
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=X.columns,
    class_names=np.unique(y).astype(str),
    mode='classification'
)

i = np.random.randint(0, X_test.shape[0])
exp = explainer_lime.explain_instance(X_test[i], model.predict_proba)
exp.show_in_notebook(show_table=True, show_all=False)
