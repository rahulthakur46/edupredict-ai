<div align="center">

# 🎓 EduPredict AI
### Student Academic Performance Prediction System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-KNN-orange?style=for-the-badge&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An intelligent ML-powered web application to predict student academic performance using the xAPI Edu Dataset.**

---

*Created with ❤️ by **RAHUL THAKUR***

</div>

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Live Demo](#-live-demo)
- [App Features](#-app-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Model Details](#-model-details)
- [Project Structure](#-project-structure)
- [Local Setup](#-local-setup)
- [Streamlit Cloud Deployment](#️-streamlit-cloud-deployment)
- [Screenshots](#-screenshots)
- [Author](#-author)

---

## 🧠 About the Project

**EduPredict AI** is a full-featured Streamlit web application that uses a **K-Nearest Neighbors (KNN)** machine learning model to predict student academic performance based on behavioral, demographic, and engagement data.

Students are classified into **3 performance bands**:

| Class | Label | Meaning |
|:---:|---|---|
| 🏆 **H** | High Performer | Strong engagement & results |
| 📘 **M** | Mid-Level | Average performance |
| ⚠️ **L** | Needs Support | Low engagement, at-risk student |

This tool empowers educators and institutions to **proactively identify students who need help** before it's too late.

---

## 🌐 Live Demo

> 🚀 **[Click here to open the app](https://your-app-link.streamlit.app)**
> *(Replace with your Streamlit Cloud URL after deployment)*

---

## ✨ App Features

| Page | Icon | Description |
|---|:---:|---|
| **Home** | 🏠 | KPI cards, class distribution pie chart, engagement bar chart, how-it-works guide |
| **Predict** | 🔮 | Full student input form → real-time KNN prediction with confidence % + radar chart |
| **Analytics** | 📊 | 15+ interactive Plotly charts — distributions, academics, family, correlations |
| **Dataset Explorer** | 📋 | Filter, browse, and download the full xAPI dataset as CSV |
| **About** | ℹ️ | Model info, feature list, tech stack, creator credits |

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **Streamlit** | Web app framework |
| **Scikit-learn** | KNN model training & prediction |
| **Joblib** | Model serialization / loading |
| **Pandas & NumPy** | Data manipulation |
| **Plotly Express** | Interactive visualizations |

---

## 📂 Dataset

- **Name:** xAPI-Edu-Data
- **Source:** [Kaggle — xAPI Educational Mining](https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data)
- **Records:** 480 students
- **Features:** 17 columns (16 input features + 1 target)

### Key Features Used:

| Feature | Type | Description |
|---|---|---|
| `raisedhands` | Numeric | Times student raised hand in class |
| `VisITedResources` | Numeric | Times student visited course content |
| `AnnouncementsView` | Numeric | Times student checked announcements |
| `Discussion` | Numeric | Times student participated in discussions |
| `gender` | Categorical | M / F |
| `NationalITy` | Categorical | Student's nationality |
| `StageID` | Categorical | School stage (Lower / Middle / High) |
| `Topic` | Categorical | Subject enrolled in |
| `StudentAbsenceDays` | Categorical | Above-7 / Under-7 |
| `ParentAnsweringSurvey` | Categorical | Yes / No |
| `ParentschoolSatisfaction` | Categorical | Good / Bad |
| *(+ more...)* | | |

---

## 🤖 Model Details

| Parameter | Value |
|---|---|
| **Algorithm** | K-Nearest Neighbors (KNN) |
| **K (neighbors)** | 7 |
| **Encoding** | One-Hot Encoding for all categorical features |
| **Input Features** | 72 (after encoding) |
| **Output Classes** | L (0) · M (1) · H (2) |
| **Model File** | `knn_model__2_.pkl` |
| **Library** | Scikit-learn 1.6.1 |

---

## 📁 Project Structure

```
📦 edupredict-ai/
├── 📄 app.py                  # Main Streamlit application
├── 🤖 knn_model__2_.pkl       # Trained KNN model
├── 📊 xAPI-Edu-Data.csv       # Dataset
├── 📋 requirements.txt        # Python dependencies
└── 📖 README.md               # This file
```

---

## 🚀 Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/edupredict-ai.git
cd edupredict-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## ☁️ Streamlit Cloud Deployment

### Step 1 — Push to GitHub
Make sure all 4 files are in your GitHub repository:
```
app.py
knn_model__2_.pkl
xAPI-Edu-Data.csv
requirements.txt
```

### Step 2 — Deploy
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository and set **Main file:** `app.py`
5. Click **"Deploy"** 🚀

Your app will be live at:
```
https://your-app-name.streamlit.app
```

---

## 📸 Screenshots

> *(Add screenshots of your app here after deployment)*

| Home Dashboard | Prediction Page |
|---|---|
| ![Home](screenshots/home.png) | ![Predict](screenshots/predict.png) |

| Analytics | Dataset Explorer |
|---|---|
| ![Analytics](screenshots/analytics.png) | ![Dataset](screenshots/dataset.png) |

---

## 👨‍💻 Author

<div align="center">

### RAHUL THAKUR

*Data Scientist & ML Developer*

Built with ❤️ using Python, Streamlit, Scikit-learn & Plotly

---

⭐ **If you found this project useful, please give it a star!** ⭐

</div>