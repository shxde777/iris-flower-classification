# 🌸 Iris Flower Classification — Web Application

A clean, professional **Streamlit** web app that predicts Iris flower species using a **Random Forest** machine learning model.

Built as an internship-ready project demonstrating end-to-end ML workflow: data loading → model training → web deployment.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Project Overview

The [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) is a classic beginner ML dataset containing **150 samples** across **3 species**, with **4 features** each (sepal length, sepal width, petal length, petal width).

This project trains a **Random Forest Classifier** on the dataset and serves predictions through an interactive Streamlit web interface — users adjust sliders and instantly see the predicted species along with confidence scores.

---

## ✨ Features

- 🔮 **Real-time Prediction** — Predict Iris species instantly via slider inputs
- 📊 **Confidence Scores** — Visual confidence breakdown for all three species
- 🔬 **Feature Importance** — See which measurements matter most to the model
- 🗂️ **Dataset Explorer** — Browse the full Iris dataset within the app
- 🎨 **Modern Dark UI** — Professional gradient design with Inter font
- 📱 **Responsive Layout** — Works on desktop and mobile browsers

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core language |
| **Streamlit** | Web application framework |
| **Scikit-learn** | Machine learning (Random Forest) |
| **Pandas** | Data manipulation & display |
| **NumPy** | Numerical computing |
| **Joblib** | Model serialization |

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/iris-flower-classification.git
cd iris-flower-classification

# 2. (Optional) Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (run once)
python train_model.py

# 5. Launch the app
streamlit run app.py
```

> Replace `YOUR_USERNAME` with your actual GitHub username.

The app will open at **http://localhost:8501**.

---

## ▶️ Quick Commands

| Action | Command |
|--------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Train model | `python train_model.py` |
| Run app | `streamlit run app.py` |

---

## 📁 Folder Structure

```
iris-flower-classification/
│
├── app.py                # Streamlit web application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation (this file)
├── .gitignore             # Git ignore rules
│
└── model/
    └── iris_model.pkl     # Trained model (auto-generated)
```

---

## 🔮 Future Improvements

- [ ] Add more classifiers (SVM, KNN, Logistic Regression) for comparison
- [ ] Interactive scatter plots and pair plots
- [ ] CSV upload for batch predictions
- [ ] Deploy to Streamlit Cloud
- [ ] Add unit tests
- [ ] Dark / Light theme toggle

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Scikit-learn](https://scikit-learn.org/) — Iris dataset & ML tools
- [Streamlit](https://streamlit.io/) — Web framework
- [Ronald Fisher](https://en.wikipedia.org/wiki/Ronald_Fisher) — Original dataset (1936)

---

<p align="center">
  Built with ❤️ using <strong>Streamlit</strong> & <strong>Scikit-learn</strong>
</p>
