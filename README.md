# 🏥 EMR Readmission Prediction

This project predicts **30-day hospital readmissions** using **Electronic Medical Records (EMR)** data.  
It combines **Machine Learning (Random Forest)** with a **Streamlit web app** to provide predictions in a simple interface for healthcare staff and researchers.

---

## 📌 Features
- Clean EMR preprocessing pipeline (handling missing values, encoding, SMOTE)
- Random Forest classifier (Accuracy: 94%, Precision: 97%, Recall: 90%)
- Interactive **Streamlit** app for uploading CSV files and getting predictions
- Modular design (`Preprocessor`, `ModelHandler`, `StreamlitApp`) for easy upgrades
- Future scope: chatbot interface + NLP on clinical notes

---

## 📂 Project Structure
```
EMR-Chatbot/
│
├── app/
│   └── streamlit_app.py        # Streamlit front-end
│
├── utils/
│   └── preprocess.py           # Data preprocessing logic
│
├── models/
│   └── readmission_model.pkl   # Pre-trained model
│   └── column_names.pkl        # Feature list
│
├── data/
│   └── diabetic_data.csv       # Dataset (not uploaded, link provided below)
│
├── notebooks/
│   └── train_model.ipynb       # Training notebook
│
├── requirements.txt            # Dependencies
├── train_model.py              # Training script
└── README.md                   # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/emr-readmission-prediction.git
cd emr-readmission-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download from Kaggle:  
👉 [Diabetes 130-US hospitals dataset](kaggle.com/code/chongchong33/predicting-hospital-readmission-of-diabetics/input?select=diabetic_data.csv)

Place it in:
```
data/diabetic_data.csv
```

### 4. Run the App
```bash
streamlit run app/streamlit_app.py
```
Then open `http://localhost:8501` in your browser.

---

## 🎮 Usage
1. Upload a `.csv` file with patient EMR records.  
2. The app will preprocess and align columns.  
3. Predictions appear instantly with both numeric (`0/1`) and label form.  

Example output:

| Age  | Gender | Medications | Readmission_Predicted | Readmission_Label      |
|------|--------|-------------|------------------------|------------------------|
| 60-70| Male   | 12          | 1                      | Readmitted (<30 Days)  |
| 40-50| Female | 5           | 0                      | Not Readmitted         |

---

## 📓 Retrain Model
To retrain:
```bash
python train_model.py
```
or use the Jupyter notebook `notebooks/train_model.ipynb`.

This will create:
- `models/readmission_model.pkl`
- `models/column_names.pkl`

---

## 🚀 Deployment
You can deploy on:
- [Streamlit Cloud](https://streamlit.io/cloud) (free)
- [Hugging Face Spaces](https://huggingface.co/spaces)

---

## 🔮 Future Enhancements
- Conversational chatbot integration
- NLP on clinical notes
- Multi-disease prediction
- Cloud / hospital system integration
- HIPAA/GDPR compliance

---

## 📜 License
For educational purposes only.  
Do not use with real patient data without proper compliance.
