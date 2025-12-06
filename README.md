# News Article Topic Classification 📰🤖

End-to-end machine learning project for classifying news articles into topics using the AG News dataset from HuggingFace.

## 🎯 Project Overview

This project implements a complete ML pipeline following CRISP-DM methodology:
- **Dataset:** AG News (120K training samples, 4 categories)
- **Models:** Baseline (TF-IDF + Logistic Regression) and Transformer (DistilBERT)
- **Tracking:** MLflow for experiment management
- **Deployment:** Streamlit web application

## 📊 Dataset: AG News

- **Source:** [HuggingFace ag_news](https://huggingface.co/datasets/ag_news)
- **Classes:**
  - 0: World
  - 1: Sports
  - 2: Business
  - 3: Sci/Tech
- **Train:** 120,000 samples
- **Test:** 7,600 samples

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/darshlukkad/News-Article-classification.git
cd News-Article-classification

# Install dependencies
pip install -r requirements.txt
```

### Training Models

```bash
# Train baseline model (TF-IDF + Logistic Regression)
python src/train_baseline.py

# Train transformer model (DistilBERT)
python src/train_transformer.py
```

### View Experiments

```bash
# Launch MLflow UI
mlflow ui

# Open browser at http://localhost:5000
```

### Run Web App

```bash
# Launch Streamlit app
streamlit run app/streamlit_app.py

# Open browser at http://localhost:8501
```

## 📁 Project Structure

```
news-topic-classification/
├── data/                       # Dataset cache (gitignored)
├── notebooks/
│   └── 01_training.ipynb      # Complete training pipeline
├── src/
│   ├── __init__.py
│   ├── data_prep.py           # Data loading & preprocessing
│   ├── train_baseline.py      # Baseline model training
│   ├── train_transformer.py   # Transformer training
│   ├── evaluate.py            # Evaluation metrics
│   └── inference.py           # Prediction utilities
├── app/
│   └── streamlit_app.py       # Web application
├── models/                     # Saved models (gitignored)
├── mlruns/                     # MLflow tracking (gitignored)
├── figures/                    # Visualizations
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── crisp_dm.md                # CRISP-DM documentation
```

## 🎓 Documentation

- **[CRISP-DM Documentation](crisp_dm.md)** - Detailed methodology
- **[Slides Outline](slides_outline.md)** - Presentation structure
- **[Demo Script](demo_script.md)** - Video walkthrough guide

## 📈 Results

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| Baseline (TF-IDF + LR) | TBD | TBD | ~2 min |
| Transformer (DistilBERT) | TBD | TBD | ~30 min |

*Results will be updated after training*

## 🛠️ Technology Stack

- **Python 3.8+**
- **scikit-learn** - Baseline model
- **PyTorch & Transformers** - Deep learning
- **MLflow** - Experiment tracking
- **Streamlit** - Web deployment
- **HuggingFace Datasets** - Data loading

## 📝 License

MIT License

## 👨‍💻 Author

Darsh Lukkad - [GitHub](https://github.com/darshlukkad)

## 🙏 Acknowledgments

- AG News dataset from HuggingFace
- DistilBERT model from HuggingFace Transformers
- MLflow for experiment tracking
