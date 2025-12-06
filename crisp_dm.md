# CRISP-DM Documentation: News Article Topic Classification

## 📋 Table of Contents
1. [Business Understanding](#1-business-understanding)
2. [Data Understanding](#2-data-understanding)
3. [Data Preparation](#3-data-preparation)
4. [Modeling](#4-modeling)
5. [Evaluation](#5-evaluation)
6. [Deployment](#6-deployment)

---

## 1. Business Understanding

### 1.1 Business Objectives
- **Goal:** Automate the categorization of news articles into predefined topics
- **Stakeholders:** News aggregators, content management platforms, media companies
- **Business Value:** 
  - Reduce manual tagging time by 80%
  - Enable real-time content categorization
  - Improve content discoverability and user experience
  - Scale content processing capabilities

### 1.2 Success Criteria
- **Primary Metric:** F1-score > 0.85 on test set
- **Secondary Metrics:**
  - Accuracy > 85%
  - Inference time < 100ms per article
  - Model size < 500MB for deployment
- **Business Impact:** Process 10,000+ articles per day with minimal human intervention

### 1.3 Project Plan
- **Timeline:** 48 hours (proof of concept)
- **Deliverables:**
  - Trained baseline and transformer models
  - MLflow experiment tracking setup
  - Streamlit web application
  - Complete documentation

---

## 2. Data Understanding

### 2.1 Data Collection
- **Source:** AG News dataset from HuggingFace
- **Access:** `datasets.load_dataset("ag_news")`
- **License:** Open source, freely available for research and commercial use

### 2.2 Data Description
- **Size:**
  - Training: 120,000 samples (30,000 per class)
  - Testing: 7,600 samples (1,900 per class)
- **Features:**
  - `text`: Article title + description (string)
  - `label`: Category (integer 0-3)
- **Target Classes:**
  - 0: World
  - 1: Sports
  - 2: Business
  - 3: Sci/Tech

### 2.3 Exploratory Data Analysis
*To be completed during data preparation:*
- Text length distribution
- Vocabulary size and most frequent words
- Class distribution (balanced dataset)
- Sample articles from each category
- Missing values check

### 2.4 Data Quality
- **Completeness:** No missing values expected
- **Consistency:** Pre-cleaned and standardized format
- **Class Balance:** Perfectly balanced (25% per class)
- **Potential Issues:**
  - Text length variation
  - Domain-specific terminology
  - Temporal drift (news from specific time period)

---

## 3. Data Preparation

### 3.1 Data Selection
- Use complete training set (120K samples)
- Use complete test set (7.6K samples)
- No additional filtering required

### 3.2 Data Cleaning
**For Baseline Model (TF-IDF):**
- Convert text to lowercase
- Remove special characters and punctuation
- Remove extra whitespace
- Optional: Remove stopwords

**For Transformer Model (DistilBERT):**
- Minimal cleaning (preserve original text structure)
- BERT tokenizer handles preprocessing

### 3.3 Feature Engineering

**Baseline Model:**
- TF-IDF vectorization (max 5000 features)
- N-grams: unigrams and bigrams
- Min document frequency: 2
- Max document frequency: 0.95

**Transformer Model:**
- Tokenization using DistilBERT tokenizer
- Max sequence length: 128 tokens
- Padding and truncation
- Attention masks

### 3.4 Data Splitting
- **Training Set:** 120,000 samples (pre-split by HuggingFace)
- **Test Set:** 7,600 samples (pre-split by HuggingFace)
- **Validation:** Use 10% of training set during transformer training

---

## 4. Modeling

### 4.1 Modeling Technique Selection

**Model 1: Baseline - TF-IDF + Logistic Regression**
- **Rationale:** Fast, interpretable, good benchmark
- **Advantages:** Low computational cost, explainable predictions
- **Limitations:** Cannot capture context and word order

**Model 2: Transformer - DistilBERT**
- **Rationale:** State-of-the-art NLP, captures semantic meaning
- **Advantages:** High accuracy, transfer learning, contextual understanding
- **Limitations:** Higher computational cost, longer training time

### 4.2 Model Building

**Baseline Model Specifications:**
```python
TF-IDF Parameters:
- max_features: 5000
- ngram_range: (1, 2)
- min_df: 2
- max_df: 0.95

Logistic Regression:
- solver: 'lbfgs'
- max_iter: 1000
- multi_class: 'multinomial'
```

**Transformer Model Specifications:**
```python
Model: distilbert-base-uncased
Fine-tuning Parameters:
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Optimizer: AdamW
- Scheduler: Linear warmup
- Max sequence length: 128
```

### 4.3 MLflow Tracking

**Logged Parameters:**
- Model type (baseline/transformer)
- Hyperparameters (learning rate, batch size, etc.)
- Feature engineering settings

**Logged Metrics:**
- Training/validation loss
- Accuracy
- F1-score (macro and micro)
- Precision and recall per class

**Logged Artifacts:**
- Trained model files
- Confusion matrices
- Classification reports
- Training curves

---

## 5. Evaluation

### 5.1 Evaluation Metrics

**Primary Metrics:**
- **Accuracy:** Overall correct predictions
- **F1-Score (Macro):** Average F1 across all classes
- **F1-Score (Micro):** Overall F1 weighted by support

**Per-Class Metrics:**
- Precision, Recall, F1-score for each category
- Confusion matrix to identify misclassifications

### 5.2 Model Comparison

| Aspect | Baseline | Transformer |
|--------|----------|-------------|
| Accuracy | TBD | TBD |
| F1-Score (Macro) | TBD | TBD |
| Training Time | ~2 minutes | ~30 minutes |
| Inference Time | <10ms | ~50ms |
| Model Size | <100MB | ~250MB |

### 5.3 Error Analysis
*To be completed after training:*
- Which classes are most confused?
- Common misclassification patterns
- Sample misclassified articles
- Potential improvements

### 5.4 Model Selection
- **Production Model:** Transformer (DistilBERT) if F1 > 0.90
- **Fallback:** Baseline if resource constraints
- **Criteria:** Balance between accuracy and deployment cost

---

## 6. Deployment

### 6.1 Deployment Strategy

**Environment:**
- Web application using Streamlit
- Model served from MLflow registry or saved checkpoint
- Containerization option: Docker (future enhancement)

**Infrastructure:**
- Local deployment for POC
- Cloud deployment options: AWS/Azure/GCP (future)
- API endpoint option: FastAPI (future enhancement)

### 6.2 Deployment Implementation

**Streamlit Application Features:**
- Text input for news article
- Real-time prediction
- Confidence scores for all classes
- Model selection (baseline vs transformer)

**Model Loading:**
```python
# Load from MLflow
model = mlflow.pyfunc.load_model(model_uri)

# Or load from saved checkpoint
model = torch.load('models/best_model.pth')
```

### 6.3 Monitoring and Maintenance

**Performance Monitoring:**
- Track inference time
- Monitor prediction distribution
- Log user feedback (if available)

**Model Updates:**
- Retrain quarterly with new data
- Track model drift
- A/B testing for model improvements

### 6.4 User Guide

**For End Users:**
1. Open Streamlit app
2. Enter news article text
3. Click "Classify"
4. View predicted category and confidence

**For Developers:**
1. Clone repository
2. Install dependencies
3. Train models or load pre-trained
4. Launch Streamlit app

---

## 📊 Summary

This CRISP-DM documentation outlines the complete methodology for building an automated news article classification system using the AG News dataset. The project follows industry best practices and delivers:

✅ Clear business objectives and success criteria  
✅ Thorough data understanding and preparation  
✅ Two complementary modeling approaches  
✅ Comprehensive evaluation framework  
✅ Production-ready deployment solution  

**Next Steps:**
- Execute training pipeline
- Complete evaluation with actual results
- Deploy and test Streamlit application
- Gather user feedback for improvements
