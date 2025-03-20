## 🫀 AI-Powered Early Detection of Cardiovascular Diseases
This repository contains an AI-driven ECG classification model that detects cardiovascular abnormalities such as arrhythmia and atrial fibrillation using a hybrid CNN-LSTM deep learning approach.

## 📌 Project Overview
Cardiovascular diseases (CVDs) are a leading cause of mortality worldwide. Early and accurate detection using ECG (Electrocardiography) signals can significantly improve patient outcomes. This project leverages deep learning to automate ECG signal analysis, reducing human error and diagnostic delays in medical practice.

🔹 Dataset Used: MIT-BIH Arrhythmia Database
🔹 Model Type: CNN-LSTM (Hybrid Deep Learning Model)
🔹 Performance: 98.4% Accuracy
🔹 Deployment: Cloud API & Mobile App Compatibility

## ⚡ Features
✅ Automated ECG Signal Processing - No manual interpretation needed.
✅ Hybrid CNN-LSTM Model - Combines spatial and temporal analysis.
✅ Real-Time Diagnosis - Optimized for cloud and edge deployment.
✅ Noise Reduction & Feature Extraction - Preprocessing techniques enhance model accuracy.
✅ Complies with Medical Standards - Aligns with AAMI EC57, FDA & HIPAA regulations.

## 🏥 How It Works?

1️⃣ Load ECG Data from the MIT-BIH Arrhythmia Database

2️⃣ Preprocess Data: Noise removal, heartbeat segmentation, feature extraction

3️⃣ Train AI Model: CNN extracts features, LSTM captures temporal dependencies

4️⃣ Evaluate Performance: Accuracy, precision, recall, and AUC-ROC analysis

5️⃣ Deploy the Model: Cloud API or embedded system for real-time ECG analysis


## 🚀 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/akshaypatra/Cardiovascular_Disease_Detection.git

cd Cardiovascular_Disease_Detection

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Model

python model.py


## 📊 Results & Performance

✅ Model Accuracy: 98.4%
✅ Precision & Recall: Evaluated per class (Normal, AFib, Arrhythmia)
✅ AUC-ROC Score: High separability between normal and diseased ECG signals

## 📁 Dataset Information

Source: MIT-BIH Arrhythmia Database
Total Samples: 48 ECG recordings, 650,000+ heartbeats
Annotations: Normal beats, Arrhythmia, Atrial Fibrillation

## 🛠 Technologies Used

🔹 Python, TensorFlow, Keras - Deep Learning Model

🔹 Scikit-learn, Pandas, NumPy - Data Processing & Preprocessing

🔹 Matplotlib, Seaborn - Visualization

🔹 WFDB Library - ECG Signal Handling


