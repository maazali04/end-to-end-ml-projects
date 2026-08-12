# 🚢 Titanic Survival Predictor

A full-stack machine learning application that predicts passenger survival on the Titanic using passenger demographics, ticket information, and cabin class.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Vercel](https://img.shields.io/badge/Vercel-000000?style=flat&logo=vercel&logoColor=white)](https://vercel.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌐 Live Demo

**Frontend — Streamlit**

👉 [Open the Titanic Survival Predictor](https://maazali04-titanic-app.streamlit.app)

**Backend — FastAPI**

👉 [Open the API Documentation](https://maazali04-titanic-api.vercel.app/docs)

---

## 📌 About

This project demonstrates an end-to-end machine learning workflow, from model development and evaluation to deployment and application integration.

The trained Scikit-Learn pipeline is served through a FastAPI REST API, while a Streamlit application provides the user interface.

```text
User
  ↓
Streamlit
  ↓
FastAPI REST API
  ↓
Scikit-Learn Pipeline
  ↓
Prediction
  ↓
Streamlit
```

This project is part of my [`end-to-end-ml-projects`](https://github.com/maazali04/end-to-end-ml-projects) repository and is located in:

[`01-titanic-survival-prediction`](https://github.com/maazali04/end-to-end-ml-projects/tree/main/01-titanic-survival-prediction)

---

## 🧠 Machine Learning

The iterative development and experimentation are fully documented in the project notebook:

👉 **[`notebooks/01_eda_and_modeling.ipynb`](https://github.com/maazali04/end-to-end-ml-projects/blob/main/01-titanic-survival-prediction/notebooks/01_eda_and_modeling.ipynb)**

The notebook covers:
- Exploratory Data Analysis (EDA)
- Data preprocessing and missing value imputation
- Feature engineering
- Model experimentation and evaluation
- Final model selection
- Prediction generation and Kaggle submission

The final preprocessing and model workflow is saved as a Scikit-Learn pipeline and serialized using `joblib`.

---

## ⚙️ Tech Stack

| Area | Technologies |
|---|---|
| Machine Learning | Python, Pandas, NumPy, Scikit-Learn |
| Model Serialization | Joblib |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit |
| Deployment | Vercel, Streamlit Community Cloud |
| Containers | Docker, Docker Compose, Dev Containers |
| Version Control | Git, GitHub |

---

## 🏗️ Architecture

### Frontend

The Streamlit application collects passenger information and sends it to the FastAPI backend as a JSON request.

### Backend

FastAPI validates the request with Pydantic, passes the data through the trained Scikit-Learn pipeline, and returns the prediction.

### Deployment

```text
                         GitHub
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
      Streamlit Cloud                 Vercel
              ↓                         ↓
       Streamlit App             FastAPI Backend
                                        ↓
                              Titanic ML Pipeline
                                        ↓
                                   Prediction
```

---

## 🚀 Run Locally

### Clone the repository

```bash
git clone https://github.com/maazali04/end-to-end-ml-projects.git
cd end-to-end-ml-projects/01-titanic-survival-prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the FastAPI backend

```bash
uvicorn api.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### Start the Streamlit application

```bash
streamlit run app.py
```

---

## 🧪 API Testing

The project includes a test script for checking communication with the FastAPI backend.

```bash
python test_api.py
```

You can also test the deployed API using the interactive Swagger documentation:

👉 [FastAPI API Docs](https://maazali04-titanic-api.vercel.app/docs)

---

## 📂 Project Location

This project is part of the main repository:

**End-to-End ML Projects**

👉 [GitHub Repository](https://github.com/maazali04/end-to-end-ml-projects)

**Titanic Survival Prediction**

👉 [Project Directory](https://github.com/maazali04/end-to-end-ml-projects/tree/main/01-titanic-survival-prediction)

---

## 👨‍💻 Author

| **Maaz Ali** |
| :--- |
| **Focus:** Machine Learning, FastAPI, Streamlit, Docker & Cloud Deployment |
| **GitHub:** [@maazali04](https://github.com/maazali04) |
| **Kaggle:** [@maazali04](https://www.kaggle.com/maazali04) |
| **Main Repository:** [`end-to-end-ml-projects`](https://github.com/maazali04/end-to-end-ml-projects) |

*Built as part of an ongoing series to master practical full-stack machine learning workflows.*

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

Copyright © 2026 Maaz Ali.

---

⭐ If you find this project useful, feel free to explore the repository and check out the other machine learning projects.
