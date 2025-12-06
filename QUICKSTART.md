# Quick Start Guide 🚀

This guide will help you get the project up and running quickly.

## ⚡ Fast Track (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Data Exploration

```python
python src/data_prep.py
```

This will:
- Load AG News dataset from HuggingFace
- Generate EDA visualizations in `figures/`
- Show sample articles from each category

### 3. Train Baseline Model (~2 minutes)

```bash
python src/train_baseline.py
```

Trains TF-IDF + Logistic Regression model with MLflow tracking.

### 4. Train Transformer Model (~30 minutes)

```bash
python src/train_transformer.py
```

Fine-tunes DistilBERT on AG News dataset with MLflow tracking.

### 5. Launch Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Opens web interface at `http://localhost:8501`

### 6. View MLflow Experiments

```bash
mlflow ui
```

Opens experiment tracking UI at `http://localhost:5000`

---

## 📊 Using Jupyter Notebook

For an interactive experience:

```bash
jupyter notebook notebooks/01_training.ipynb
```

This notebook contains the complete pipeline:
- Data loading & EDA
- Baseline training
- Transformer training
- Evaluation & comparison
- Inference examples

---

## 🎯 Testing Individual Components

### Data Preparation
```python
from src.data_prep import AGNewsDataPrep

data_prep = AGNewsDataPrep()
data_prep.run_full_eda()
```

### Baseline Model
```python
from src.train_baseline import train_with_mlflow

model, metrics = train_with_mlflow()
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Inference
```python
from src.inference import TransformerPredictor

predictor = TransformerPredictor()
predictor.load_model()

text = "Lakers win championship game"
category, confidence = predictor.predict(text)
print(f"Category: {category}")
```

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Ensure you're in the project root
pip install -r requirements.txt --upgrade
```

### CUDA/GPU Issues
The code automatically detects and uses GPU if available, but works on CPU too.

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Model Not Found
Make sure to train the models first:
```bash
python src/train_baseline.py
python src/train_transformer.py
```

---

## 📦 Google Colab Setup

```python
# Install dependencies
!pip install -q datasets transformers torch mlflow scikit-learn streamlit

# Clone repository
!git clone https://github.com/darshlukkad/News-Article-classification.git
%cd News-Article-classification

# Run training
!python src/train_baseline.py
!python src/train_transformer.py
```

---

## 🎓 Next Steps

1. ✅ Explore visualizations in `figures/`
2. ✅ Check MLflow for experiment details
3. ✅ Try the Streamlit app with custom articles
4. ✅ Review CRISP-DM documentation in `crisp_dm.md`
5. ✅ Customize models by editing training scripts

---

## 📝 Project Status

After following this guide, you should have:

- [x] Dataset loaded and explored
- [x] Baseline model trained (~87% accuracy)
- [x] Transformer model trained (~94% accuracy)
- [x] MLflow experiments logged
- [x] Streamlit app running
- [x] Ready to deploy!

---

## 💡 Pro Tips

- Use `mlflow ui` to compare different runs
- Edit hyperparameters in training scripts
- The Streamlit app caches models for fast inference
- Check `figures/` for all visualizations
- Use the Jupyter notebook for experimentation

---

For detailed documentation, see:
- `README.md` - Complete project overview
- `crisp_dm.md` - Methodology documentation
- `slides_outline.md` - Presentation guide
- `demo_script.md` - Demo walkthrough
