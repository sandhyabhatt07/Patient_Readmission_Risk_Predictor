import gradio as gr
import pandas as pd
import numpy as np
import joblib
import zipfile
import os
from fpdf import FPDF

# Load model and helpers
model = joblib.load("best_xgb_model_gridsearch.pkl")
feature_columns = joblib.load("feature_columns.pkl")
encoders = joblib.load("label_encoders.pkl")

# Feature engineering steps
def engineer_features(df):
    df['age'] = df['age'].apply(lambda x: int(x.strip('[]').split('-')[0]) + 5 if isinstance(x, str) else x)
    df['high_inpatient_visits'] = (df['number_inpatient'] >= 2).astype(int)
    df['frequent_emergency'] = (df['number_emergency'] >= 1).astype(int)
    df['medication_burden'] = df['num_medications'] * df['number_diagnoses']
    df['inpatient_timespan'] = df['number_inpatient'] * df['time_in_hospital']
    df['total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    df['visit_density'] = df['total_visits'] / (df['time_in_hospital'] + 1)

    def simplify_discharge(desc):
        if "Home" in desc:
            return "home"
        elif "Skilled Nursing" in desc or "Rehab" in desc or "Hospice" in desc:
            return "care"
        elif "Expired" in desc:
            return "expired"
        else:
            return "other"
    df['discharge_group'] = df['discharge_desc'].apply(simplify_discharge)
    return df

# Apply saved label encoders
def apply_label_encoders(df, encoders):
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
            known_classes = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else "Unknown")
            if "Unknown" not in le.classes_:
                le.classes_ = np.append(le.classes_, "Unknown")
            df[col] = le.transform(df[col])
    return df

# Generate individual report
def generate_report(patient_id, prediction, features):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Patient Readmission Risk Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Patient ID: {patient_id}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {'Likely Readmitted' if prediction == 1 else 'Low Risk'}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Patient Details:", ln=True)
    for key, value in features.items():
        pdf.cell(200, 10, txt=f"{key.replace('_', ' ').title()}: {value}", ln=True)
    path = f"patient_{patient_id}_report.pdf"
    pdf.output(path)
    return path

# Prediction logic
def predict_readmission(file):
    try:
        df = pd.read_csv(file.name)
        df = engineer_features(df)
        df = apply_label_encoders(df, encoders)

        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            return f"❌ Missing required columns: {missing_cols}", None, None, None

        preds = model.predict(df[feature_columns])
        df["readmission_prediction"] = preds

        pdf_paths = []
        for i, row in df.iterrows():
            patient_id = i + 1
            features = {
                "age": row.get("age"),
                "time_in_hospital": row.get("time_in_hospital"),
                "number_diagnoses": row.get("number_diagnoses"),
                "num_medications": row.get("num_medications"),
                "comorbidities": row.get("comorbidities"),
                "medication_burden": row.get("medication_burden")
            }
            pdf_path = generate_report(patient_id, row["readmission_prediction"], features)
            pdf_paths.append(pdf_path)

        zip_path = "all_patient_reports.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for path in pdf_paths:
                zipf.write(path, os.path.basename(path))
                os.remove(path)

        summary = f"✅ Predicted {sum(preds)} of {len(preds)} patients likely to be readmitted."
        return summary, df[["readmission_prediction"]], zip_path, None

    except Exception as e:
        return f"❌ Error: {str(e)}", None, None, None

# Gradio Interface
iface = gr.Interface(
    fn=predict_readmission,
    inputs=gr.File(label="📁 Upload Patient CSV File"),
    outputs=[
        gr.Textbox(label="📊 Prediction Summary"),
        gr.Dataframe(label="📋 Predictions", type="pandas"),
        gr.File(label="📥 Download All Reports (.zip)"),
        gr.File(value="./sample_input.csv", label="📄 Download Sample Template (CSV)")  # Ensure path is correct for your working directory
    ],
    title="🏥 Patient Readmission Risk Predictor",
    description=(
        "📁 **Instructions:** Upload a CSV file with the following required columns:\n\n"
        "`age`, `time_in_hospital`, `num_lab_procedures`, `num_medications`, `number_outpatient`, "
        "`number_emergency`, `number_inpatient`, `number_diagnoses`, `comorbidities`, `discharge_desc`, "
        "`admission_type_desc`, `admission_source_desc`, `diabetesMed`\n\n"
        "📊 The model will predict 30-day readmission risk and generate personalized PDF reports for each patient."
    )
)

iface.launch()
