# 🛡️ IoT Intrusion Detection System using Machine Learning

A Machine Learning based Intrusion Detection System (IDS) developed to detect malicious network traffic in Internet of Things (IoT) environments. This project compares multiple machine learning algorithms to identify the best-performing model for intrusion detection and improve IoT network security.

---

## 📌 Project Overview

The rapid growth of IoT devices has significantly increased cybersecurity risks. Traditional Intrusion Detection Systems (IDS) often struggle to detect modern and unknown attacks.

This project proposes a **Machine Learning-based IoT Intrusion Detection System** capable of identifying malicious network traffic with high accuracy using multiple ML algorithms.

The project includes:

- Data Collection
- Data Preprocessing
- Feature Engineering
- Feature Selection
- Model Training
- Model Evaluation
- Prediction on Unseen Data
- Performance Comparison

---

## 🎯 Objectives

- Detect malicious IoT network traffic.
- Improve intrusion detection accuracy using Machine Learning.
- Compare different Machine Learning algorithms.
- Evaluate models using multiple performance metrics.
- Identify the best-performing model for deployment.

---

## 🚀 Features

- Intrusion Detection for IoT Networks
- Supports Multiple Attack Categories
- Data Cleaning & Preprocessing
- Feature Engineering
- Feature Selection
- Machine Learning Model Comparison
- Confusion Matrix
- Classification Report
- Prediction on Unseen Network Traffic
- High Accuracy Detection

---

## 📂 Datasets

This project uses two publicly available datasets from Kaggle.

### Dataset 1 — IoT Intrusion Dataset

Source:
https://www.kaggle.com/datasets/subhajournal/iotintrusion

Description:

- IoT Network Traffic Dataset
- Multiple Attack Categories
- Approximately 47 Original Features

---

### Dataset 2 — NSL-KDD Dataset

Source:
https://www.kaggle.com/datasets/hassan06/nslkdd

Description:

- Standard Network Intrusion Detection Dataset
- Multiple Attack Types
- Used for Cross-Dataset Evaluation

---

> **Note:**  
> Due to GitHub's file size limitations, the datasets are **not included** in this repository.  
> Please download both datasets from the Kaggle links above and place them inside the **Code/** directory before running the project.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- Matplotlib
- Seaborn
- Joblib
- Visual Studio Code

---

## 🤖 Machine Learning Models

The following machine learning models were implemented and compared:

### 🌟 LightGBM

- Gradient Boosting Algorithm
- Fast Training
- High Accuracy
- Best Performing Model

---

### 🌲 Random Forest

- Ensemble Learning Algorithm
- Stable Predictions
- Good Generalization

---

### 🌳 Decision Tree

- Simple & Interpretable
- Easy to Visualize
- Used as Baseline Model

---

## 📊 Data Processing Pipeline

```
Dataset Collection
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Feature Selection
        │
        ▼
Train-Test Split
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Prediction
```

---

## 📈 Model Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

## 🏆 Results

### Dataset 1

| Model | Accuracy | Performance |
|--------|----------|-------------|
| LightGBM | **99.3%** | ⭐ Best |
| Random Forest | 99.33% | Very Good |
| Decision Tree | 99.29% | Good |

---

### Dataset 2

| Model | Accuracy | Performance |
|--------|----------|-------------|
| LightGBM | **99.0%** | ⭐ Best |
| Random Forest | 98.9% | Very Good |
| Decision Tree | 98.7% | Good |

---

## 📌 Final Conclusion

Among all the implemented Machine Learning models,

✅ **LightGBM achieved the highest overall performance on both datasets.**

It provided:

- High Accuracy
- Better F1 Score
- Fast Training
- Efficient Performance on Large Datasets

Therefore, **LightGBM was selected as the final recommended model.**

---

## 📁 Project Structure

```
IoT-Intrusion-Detection-System
│
├── Code/
│   ├── main.py
│   ├── main_dataset2.py
│   ├── predict.py
│   └── predict2.py
│
├── Results/
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/jemmiii/IoT-Intrusion-Detection-System.git
```

Move into the project folder

```bash
cd IoT-Intrusion-Detection-System
```

Install required libraries

```bash
pip install -r requirements.txt
```

Run the project

```bash
python main.py
```

---

## 🔮 Future Scope

- Real-Time Intrusion Detection
- Deep Learning Integration
- Cloud Deployment
- Live Traffic Monitoring
- Web Dashboard
- API Integration

---

## 👨‍💻 Author

### Jemin Patidar

🎓 Bachelor of Technology (Information Technology)

🏫 Manipal University Jaipur

GitHub:
https://github.com/jemmiii

---

## ⭐ Support

If you found this project useful,

⭐ **Please consider giving this repository a Star!**

It motivates me to build more Machine Learning and Cybersecurity projects.

---

## 📜 License

This project is intended for **educational and research purposes only.**
