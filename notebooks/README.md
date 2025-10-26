# 📓 Jupyter Notebooks - Model Training Documentation

This directory contains Jupyter notebooks that document the complete training process for each disease prediction model in the Multiple Disease Prediction Web Application.

## 📚 Available Notebooks

| Notebook | Disease | Dataset | Model | Status |
|----------|---------|---------|-------|--------|
| `01_PIMA_Diabetes_Training.ipynb` | Diabetes (PIMA) | ✅ Included | Ensemble (Voting/Stacking) | ✅ Complete |
| `02_Disease_Symptom_Prediction.ipynb` | 41 Diseases | ✅ Included | XGBoost Multi-class | ✅ Complete |
| `03_Heart_Disease_Prediction.ipynb` | Heart Disease | ⚠️ Template | Random Forest / SVM | 📝 Template |
| `04_Liver_Disease_Prediction.ipynb` | Liver Disease | ⚠️ Template | Random Forest / GBM | 📝 Template |
| `05_Lung_Cancer_Prediction.ipynb` | Lung Cancer | ✅ Included | Random Forest / GBM | ✅ Complete |

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install -r requirements.txt
```

**Required Libraries:**
- pandas, numpy
- scikit-learn
- xgboost, catboost
- matplotlib, seaborn
- imbalanced-learn
- joblib

### Running Notebooks

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Navigate to** `notebooks/` **directory**

3. **Open a notebook** and run cells sequentially (Shift+Enter)

## 📊 Notebook Descriptions

### 1. PIMA Diabetes Training (✅ Complete)
**File:** `01_PIMA_Diabetes_Training.ipynb`

**What it does:**
- Loads PIMA Indian Diabetes dataset (768 samples, 8 features)
- Feature engineering (interaction terms, polynomials)
- Handles class imbalance with SMOTE-Tomek
- Trains 11+ ML models with cross-validation
- Implements ensemble methods (soft voting with optimized weights, stacking)
- Generates comprehensive evaluation reports
- Saves final model to `../code/PIMA/artifacts/`

**Key Features:**
- ✅ Out-of-fold target encoding for categorical features
- ✅ Model calibration for better probability estimates
- ✅ Automated hyperparameter weight optimization
- ✅ PDF report generation with confusion matrix & ROC curves

**Output Models:**
- `final_model.joblib` / `final_model.sav`
- `preproc.joblib` (imputer, scaler, selector)
- `cv_scores.json` (metadata)

---

### 2. Disease Symptom Prediction (✅ Complete)
**File:** `02_Disease_Symptom_Prediction.ipynb`

**What it does:**
- Loads multi-disease symptom dataset (4931 samples, 41 diseases)
- One-hot encodes symptoms (131 unique symptoms)
- Trains XGBoost multi-class classifier
- Evaluates with classification report & confusion matrix
- Visualizes feature importance (top symptoms)
- Saves model to `../Frontend/model/`

**Key Features:**
- ✅ Handles sparse symptom data (one-hot encoding)
- ✅ Multi-class classification (41 disease classes)
- ✅ Feature importance analysis
- ✅ Interactive prediction function

**Output Models:**
- `xgboost_model.json`
- `model_binary.dat.gz` (compressed)
- `label_encoder.joblib`

---

### 3. Heart Disease Prediction (📝 Template)
**File:** `03_Heart_Disease_Prediction.ipynb`

**Status:** Template notebook - requires dataset to run

**Expected Dataset:**
- Source: [UCI Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
- Target: 0 (no disease) / 1 (disease)

**To Complete:**
1. Download heart disease dataset
2. Update dataset path in cell 2
3. Uncomment all cells
4. Run notebook end-to-end

---

### 4. Liver Disease Prediction (📝 Template)
**File:** `04_Liver_Disease_Prediction.ipynb`

**Status:** Template notebook - requires dataset to run

**Expected Dataset:**
- Source: [Indian Liver Patient Dataset (ILPD)](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records)
- Features: Age, Gender, Total Bilirubin, Direct Bilirubin, Alkaline Phosphotase, Alamine Aminotransferase, Aspartate Aminotransferase, Total Proteins, Albumin, A/G Ratio
- Target: 1 (liver patient) / 2 (non-patient)

**To Complete:**
1. Download ILPD dataset
2. Update dataset path in load cell
3. Uncomment all cells
4. Run notebook

---

### 5. Lung Cancer Prediction (✅ Complete)
**File:** `05_Lung_Cancer_Prediction.ipynb`

**What it does:**
- Loads lung cancer dataset from `Frontend/data/lung_cancer.csv`
- Encodes categorical features (Gender, symptoms)
- Trains Random Forest & Gradient Boosting models
- Compares models and selects best performer
- Visualizes feature importance
- Saves model and encoders

**Key Features:**
- ✅ Complete EDA with visualizations
- ✅ Multi-model comparison
- ✅ Feature importance analysis
- ✅ Practical prediction function

**Output Models:**
- `lung_cancer_model.sav`
- `lung_cancer_encoders.sav`
- `lung_cancer_target_encoder.sav`

---

## 🔧 Creating Additional Notebooks

Want to add notebooks for **Parkinson's**, **Chronic Kidney Disease**, **Breast Cancer**, or **Hepatitis**?

### Template Structure:
1. **Import libraries** (pandas, sklearn, etc.)
2. **Load dataset** (update path)
3. **EDA** (info, describe, visualizations)
4. **Preprocessing** (handle missing values, encode categorical, scale)
5. **Train-test split** (stratified)
6. **Model training** (compare 3-5 models)
7. **Evaluation** (accuracy, confusion matrix, classification report)
8. **Save model** (joblib to `Frontend/models/`)
9. **Usage example** (prediction function)

**Copy template from:** `03_Heart_Disease_Prediction.ipynb` and adapt!

---

## 📦 Model Artifacts Location

After running notebooks, models are saved to:

```
Multiple-Disease-Prediction-Webapp/
├── code/PIMA/artifacts/          # PIMA diabetes models
│   ├── final_model.joblib
│   ├── final_model.sav
│   ├── preproc.joblib
│   └── cv_scores.json
├── Frontend/model/                # Disease symptom model
│   ├── xgboost_model.json
│   └── model_binary.dat.gz
└── Frontend/models/               # Individual disease models
    ├── diabetes_model.sav
    ├── heart_disease_model.sav
    ├── liver_model.sav
    ├── lung_cancer_model.sav
    ├── parkinsons_model.sav
    ├── chronic_model.sav
    ├── breast_cancer.sav
    └── hepititisc_model.sav
```

---

## 🐛 Troubleshooting

### Common Issues:

**1. ModuleNotFoundError: No module named 'xgboost'**
```bash
pip install xgboost catboost
```

**2. FileNotFoundError: Dataset not found**
- Verify dataset path in notebook
- Check if dataset exists in `code/PIMA/` or `Frontend/data/`
- Download missing datasets (see links above)

**3. Jupyter kernel crashes during training**
- Reduce `n_estimators` in models
- Set `n_jobs=1` instead of `-1`
- Increase system RAM or use cloud notebooks (Colab, Kaggle)

**4. Model performance is poor**
- Check class imbalance (use SMOTE/ADASYN)
- Try hyperparameter tuning (GridSearchCV)
- Verify feature scaling is applied
- Check for data leakage

---

## 📖 Additional Resources

- **PIMA Diabetes Dataset:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **Heart Disease Dataset:** https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- **Liver Disease Dataset:** https://www.kaggle.com/datasets/uciml/indian-liver-patient-records
- **Scikit-learn Documentation:** https://scikit-learn.org/
- **XGBoost Documentation:** https://xgboost.readthedocs.io/

---

## 🤝 Contributing

To add a new disease prediction notebook:

1. Copy template notebook (`03_Heart_Disease_Prediction.ipynb`)
2. Rename to `0X_DiseaseName_Prediction.ipynb`
3. Update markdown cells with disease-specific info
4. Load appropriate dataset
5. Train and evaluate models
6. Save model to `Frontend/models/`
7. Update this README with new entry

---

## 📝 License

These notebooks are part of the Multiple Disease Prediction Web Application project.

---

## ✅ Checklist for Running Notebooks

- [ ] Install all required packages (`requirements.txt`)
- [ ] Verify dataset paths
- [ ] Run cells sequentially (don't skip cells)
- [ ] Check model artifacts are saved
- [ ] Test prediction function at the end
- [ ] Verify model integrates with Streamlit frontend

**Happy Training! 🎉**
