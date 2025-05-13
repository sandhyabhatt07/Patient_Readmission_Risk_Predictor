# Diabetes Patient Readmission Risk Predictor (Experimental)

[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/yourusername/readmission-risk-predictor)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange.svg)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Gradio-3.41.2-blue.svg)](https://gradio.app/)
[![SHAP](https://img.shields.io/badge/SHAP-0.42.1-purple.svg)](https://github.com/slundberg/shap)

An experimental machine learning application that demonstrates the concept of predicting 30-day hospital readmission risk for diabetic patients, deployed as a Gradio web interface on Hugging Face Spaces. This project serves as a proof-of-concept to explore basic machine learning methods for healthcare predictions.

## 📋 Overview

This experimental application demonstrates how a gradient boosting model (XGBoost) can be used to predict whether a diabetic patient is likely to be readmitted to the hospital within 30 days after discharge. The project focuses on testing fundamental machine learning concepts rather than implementing advanced techniques. Users can upload patient data in CSV format and receive:

- Individual risk predictions for each patient
- Downloadable PDF reports for each patient
- A summary of overall readmission risk

> **Note:** This is an experimental project intended for educational purposes and to explore basic machine learning methods. It has not been clinically validated and should not be used for actual medical decision-making.

## 🔍 Features

- **Basic ML Implementation**: Uses an XGBoost classifier trained on the diabetes hospital readmission dataset
- **Batch Processing**: Analyzes multiple patient records simultaneously
- **Simple PDF Report Generation**: Creates basic individual PDF reports for each patient
- **User-Friendly Interface**: Simple upload-and-predict workflow with Gradio
- **SHAP Visualization**: Initial exploration of feature importance using SHAP
- **Experimental Design**: Focus on testing machine learning concepts rather than production-ready features

## 🧠 Model Details

The experimental model was trained using:

- **Algorithm**: Basic XGBoost Classifier with minimal hyperparameter tuning
- **Data**: Diabetes hospital readmission dataset
- **Features**: 
  - Demographics (age, gender, race)
  - Hospital metrics (time in hospital, procedures, lab tests)
  - Diabetes medications
  - Admission/discharge information
  - Comorbidities
- **Performance**: 
  - Modest ROC-AUC scores in experimental testing
  - Primary focus on concept demonstration rather than optimized performance
  
> **Future Improvement Potential:** The current implementation uses basic machine learning methods. Advanced techniques like deep learning, ensemble methods, more sophisticated feature engineering, and clinical expertise integration could significantly improve model performance.

## 📈 Model Insights & Intervention Analysis

### Feature Importance (SHAP Analysis)
Based on SHAP analysis, the most influential factors in predicting readmission are:

1.Discharge group (strong negative impact: -1.2)
2.Inpatient timespan (+0.29)
3.Visit density (+0.24)
4.Number of inpatient visits (+0.16)
5.Number of emergency visits (+0.13)
6.Time in hospital (+0.13)

The analysis reveals that while discharge characteristics significantly reduce readmission probability, metrics related to prior healthcare utilization and hospital stay duration are the strongest positive predictors of readmission risk.

### Intervention Simulation Results
The experiment included a simulation of targeted interventions for high-risk patients:

```python
# After simulating a 10% reduction in readmissions for high-risk patients:
# Original high-risk readmissions: 235
# Reduced high-risk readmissions: 211
```

- **Precision@Top20% Before Reduction**: Quantified model's ability to identify high-risk patients
- **Precision@Top20% After Reduction**: Demonstrated maintained precision after simulated intervention
- **% Flagged Correctly**: Percentage of high-risk patients correctly identified for intervention

This simulation suggests that even with a basic model, targeted interventions could potentially reduce readmission rates when focusing on patients with the highest risk scores.

### Visualization Results
The analysis included various visualizations to understand model performance:

- **Risk Score Distribution**: Compared distributions before/after intervention
- **Readmission Status**: Visualized changes in readmission counts 
- **Precision Comparison**: Assessed precision metrics before/after intervention
- **Risk vs. Inpatient Admissions**: Explored correlation between risk scores and prior admissions
- **Correlation Matrices**: Analyzed feature relationships in high-risk groups
- **Readmission Rate Changes**: Tracked reduction in overall readmission rates

These visualizations help demonstrate the potential clinical impact of using the model for targeted interventions, even with this experimental implementation.

## 🚀 Usage

1. Visit the [Hugging Face Space](https://huggingface.co/spaces/AvocadoMuffin/readmission-risk-predictor)
2. Upload a CSV file containing patient data (see sample format below)
3. View predictions and download the ZIP file containing individual patient reports

### Sample CSV Format

Your input CSV should include these columns:

```
race,gender,age,time_in_hospital,num_lab_procedures,num_procedures,num_medications,number_outpatient,number_emergency,number_inpatient,number_diagnoses,max_glu_serum,A1Cresult,metformin,repaglinide,nateglinide,chlorpropamide,glimepiride,acetohexamide,glipizide,glyburide,tolbutamide,pioglitazone,rosiglitazone,acarbose,miglitol,troglitazone,tolazamide,examide,citoglipton,insulin,glyburide-metformin,glipizide-metformin,glimepiride-pioglitazone,metformin-rosiglitazone,metformin-pioglitazone,change,diabetesMed,admission_type_desc,discharge_desc,admission_source_desc,comorbidities
```

## 🛠️ Technical Implementation

This experimental application consists of:

1. **Basic Data Preprocessing**: Simple handling of categorical variables and missing values
2. **Model Training**: XGBoost classifier with basic hyperparameter exploration
3. **Simple Feature Handling**: Basic inclusion of comorbidity data and medication information
4. **Web Interface**: Simple Gradio-based UI for demonstration purposes
5. **Basic Report Generation**: Minimal PDF reports using FPDF library

### 🔮 Potential Future Enhancements

For a more robust implementation, consider these improvements:

1. **Advanced Feature Engineering**: Incorporate medical domain knowledge and temporal features
2. **Deep Learning Models**: Explore RNNs or transformers for sequential medical data
3. **Ensemble Methods**: Combine multiple models for improved prediction accuracy
4. **Explainable AI**: More sophisticated techniques beyond basic SHAP visualizations
5. **Clinical Validation**: Collaborate with healthcare professionals for model validation
6. **Advanced UI**: More interactive and detailed reporting features

## 💻 Local Development

To run this application locally:

```bash
# Clone repository
git clone https://github.com/sandhyabhatt07/patient_readmission-risk-predictor.git
cd readmission-risk-predictor

# Install dependencies
pip install pandas scikit-learn xgboost shap fpdf gradio

# Run the application
python app.py
```

## 📚 Dataset

The model was trained on the UCI Diabetes 130-US hospitals dataset, which includes data from 1999-2008 across 130 US hospitals, focusing on diabetic encounters.

## 🔗 References

1. Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J., & Clore, J.N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.
2. XGBoost: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
3. SHAP: [https://github.com/slundberg/shap](https://github.com/slundberg/shap)


## ✨ Experiment Context

This project was developed as part of a machine learning experiment to:
- Test basic ML methods on healthcare data
- Explore the feasibility of readmission prediction
- Learn about model deployment using Gradio and Hugging Face
- Demonstrate a simple end-to-end ML workflow

The focus was on learning and experimentation rather than developing a production-ready clinical tool.


