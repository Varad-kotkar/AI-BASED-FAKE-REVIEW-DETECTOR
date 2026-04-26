
# 🧠 AI-Based Fake Review Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![License](https://img.shields.io/badge/License-Academic-lightgrey)
![Contributions](https://img.shields.io/badge/Contributors-Viraj%20Pathare,%20Shubham%20Dhawale,%20Chaitanya%20Ghangale-blueviolet)
---

## 📌 Overview

This project detects fake reviews using Machine Learning models.
It analyzes textual data and classifies reviews as **genuine** or **fake** using real datasets.

The system compares multiple models and selects the one with the highest accuracy.

---

## 🚀 Features

* Fake vs Genuine review classification
* Real dataset-based training
* Multiple ML model comparison
* Best model selection
* Confusion Matrix visualization
* Minimal and exam-friendly structure

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

---

## 📂 Project Structure

```
AI-Fake-Review-Detection/
│── dataset/
│   └── reviews.csv
│
│── main.py
│── model_training.py
│── evaluation.py
│
│── README.md
│── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/AI-Fake-Review-Detection.git
cd AI-Fake-Review-Detection
pip install -r requirements.txt
```

---

## ▶️ Run Project

```bash
python main.py
```

---

## 🧪 Machine Learning Models Used

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)
* Random Forest

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 📈 System Architecture Diagram

```mermaid
flowchart TD
    A[Dataset: Reviews.csv] --> B[Data Preprocessing]
    B --> C[Text Cleaning]
    C --> D[Feature Extraction (TF-IDF)]
    
    D --> E1[Logistic Regression]
    D --> E2[Naive Bayes]
    D --> E3[SVM]
    D --> E4[Random Forest]

    E1 --> F[Model Evaluation]
    E2 --> F
    E3 --> F
    E4 --> F

    F --> G[Accuracy Comparison]
    G --> H[Best Model Selection]
    H --> I[Confusion Matrix Output]
```

---

## 📸 Output

* Accuracy comparison of models
* Best model selection
* Confusion matrix visualization

---

## ❌ Simplified for Academic Use

* No `.pkl` model saving
* No unnecessary files
* Clean and minimal codebase

---

## 💡 Use Cases

* E-commerce platforms
* Review filtering systems
* Spam detection

---

## 👨‍💻 Authors

* Varad Rajendra Kotkar
📧 kotkarvarad12@gmail.com
Viraj Eknath Pathare
📧 pathareviraj8@gmail.com
Shubham Santosh Dhawale
📧 sdhawale380@gmail.com
Chaitanya Sudhir Ghangale
📧 chaitanyaghangale0@gmail.com

---

## 📄 License

This project is for academic and educational purposes only.

