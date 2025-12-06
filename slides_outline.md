# Presentation Slides Outline: News Article Topic Classification

## 📊 Slide Deck Structure (15-20 slides, 10-15 minutes)

---

### Slide 1: Title Slide
**Content:**
- **Title:** News Article Topic Classification with Deep Learning
- **Subtitle:** End-to-End ML Pipeline with AG News Dataset
- **Author:** Darsh Lukkad
- **Date:** December 2025
- **Visual:** Background image of news articles or neural network

---

### Slide 2: Agenda
**Content:**
1. Problem Statement & Business Context
2. Dataset Overview (AG News)
3. CRISP-DM Methodology
4. Technical Approach
5. Model Development
6. Results & Evaluation
7. Deployment & Demo
8. Conclusion & Future Work

---

### Slide 3: Problem Statement
**Content:**
- **Challenge:** Manual news categorization is time-consuming and doesn't scale
- **Impact:** News aggregators process 10,000+ articles daily
- **Solution:** Automated classification using machine learning
- **Business Value:**
  - 80% reduction in manual tagging time
  - Real-time content categorization
  - Improved content discoverability

**Visual:** Icons showing manual vs automated workflow

---

### Slide 4: AG News Dataset
**Content:**
- **Source:** HuggingFace Datasets
- **Size:** 127,600 total samples
  - Training: 120,000 (30K per class)
  - Testing: 7,600 (1.9K per class)
- **Categories:**
  - 🌍 World
  - ⚽ Sports
  - 💼 Business
  - 🔬 Sci/Tech
- **Features:** Article title + description

**Visual:** Pie chart showing class distribution (balanced)

---

### Slide 5: CRISP-DM Methodology
**Content:**
- Standard framework for data science projects
- **6 Phases:**
  1. ✅ Business Understanding
  2. ✅ Data Understanding
  3. ✅ Data Preparation
  4. ✅ Modeling
  5. ✅ Evaluation
  6. ✅ Deployment

**Visual:** CRISP-DM circular diagram

---

### Slide 6: Business Understanding
**Content:**
- **Objective:** Automate news article categorization
- **Success Metrics:**
  - F1-score > 0.85
  - Inference time < 100ms
  - Model size < 500MB
- **Stakeholders:** News platforms, content managers, readers
- **Timeline:** 48-hour POC development

**Visual:** Business value flowchart

---

### Slide 7: Data Understanding
**Content:**
- **Sample Articles:**
  - World: "UN Summit Discusses Climate Policy..."
  - Sports: "Lakers Win Championship Game..."
  - Business: "Tech Stocks Rally After Earnings..."
  - Sci/Tech: "New AI Model Breaks Records..."
- **Data Quality:** Clean, balanced, no missing values
- **Text Length:** Average 40-60 words

**Visual:** Word cloud from each category

---

### Slide 8: Technical Architecture
**Content:**
- **Pipeline Components:**
  ```
  Data Loading → Preprocessing → Model Training → 
  Evaluation → MLflow Tracking → Deployment
  ```
- **Tech Stack:**
  - Python, PyTorch, Transformers
  - MLflow for tracking
  - Streamlit for deployment
  - HuggingFace datasets

**Visual:** Architecture diagram with logos

---

### Slide 9: Two-Model Approach
**Content:**
| Aspect | Baseline | Transformer |
|--------|----------|-------------|
| **Model** | TF-IDF + Logistic Regression | DistilBERT |
| **Pros** | Fast, interpretable | High accuracy |
| **Cons** | Limited context | Resource intensive |
| **Use Case** | Quick baseline | Production model |

**Visual:** Side-by-side model comparison

---

### Slide 10: Baseline Model
**Content:**
- **Approach:** TF-IDF + Logistic Regression
- **Feature Engineering:**
  - 5,000 TF-IDF features
  - Unigrams + bigrams
  - Min/max document frequency filtering
- **Training Time:** ~2 minutes
- **Advantages:** Fast, interpretable, good benchmark

**Visual:** TF-IDF visualization or feature importance

---

### Slide 11: Transformer Model (DistilBERT)
**Content:**
- **Model:** DistilBERT (66M parameters)
- **Why DistilBERT?**
  - 40% smaller than BERT
  - 60% faster
  - 97% of BERT's performance
- **Fine-tuning:**
  - 3 epochs
  - Learning rate: 2e-5
  - Batch size: 16
- **Training Time:** ~30 minutes

**Visual:** Transformer architecture diagram

---

### Slide 12: MLflow Experiment Tracking
**Content:**
- **Logged Parameters:** Learning rate, batch size, epochs
- **Logged Metrics:** Accuracy, F1-score, loss
- **Logged Artifacts:** Models, plots, reports
- **Benefits:**
  - Reproducibility
  - Model comparison
  - Version control

**Visual:** Screenshot of MLflow UI

---

### Slide 13: Results - Performance Metrics
**Content:**
| Metric | Baseline | Transformer |
|--------|----------|-------------|
| **Accuracy** | XX.X% | XX.X% |
| **F1-Score (Macro)** | X.XX | X.XX |
| **Precision** | X.XX | X.XX |
| **Recall** | X.XX | X.XX |
| **Training Time** | 2 min | 30 min |
| **Inference Time** | <10ms | ~50ms |

**Visual:** Bar chart comparing metrics

---

### Slide 14: Confusion Matrix
**Content:**
- **Baseline Confusion Matrix** (left)
- **Transformer Confusion Matrix** (right)
- **Key Observations:**
  - Most confusion between Business and Sci/Tech
  - Sports category easiest to classify
  - Transformer reduces misclassifications

**Visual:** Two heatmap confusion matrices side by side

---

### Slide 15: Error Analysis
**Content:**
- **Common Misclassifications:**
  - Business ↔ Sci/Tech (tech company news)
  - World ↔ Business (economic policy news)
- **Example Misclassification:**
  - Article: "Apple Announces New iPhone..."
  - True: Sci/Tech
  - Predicted: Business
- **Insights:** Overlapping domains challenging

**Visual:** Sample misclassified examples

---

### Slide 16: Deployment - Streamlit App
**Content:**
- **Features:**
  - Text input for news articles
  - Real-time classification
  - Confidence scores for all categories
  - Model selection (baseline/transformer)
- **User Experience:**
  - Simple, intuitive interface
  - Fast response time
  - Visual feedback

**Visual:** Screenshot of Streamlit app

---

### Slide 17: Live Demo
**Content:**
- **Demo Steps:**
  1. Open Streamlit application
  2. Enter sample news article
  3. Click "Classify"
  4. View predictions and confidence scores
  5. Try different articles from each category

**Visual:** "LIVE DEMO" text or app screenshot

---

### Slide 18: Key Achievements
**Content:**
✅ Built complete ML pipeline in 48 hours  
✅ Achieved XX% accuracy (exceeded goal)  
✅ Implemented experiment tracking with MLflow  
✅ Deployed working web application  
✅ Documented with CRISP-DM methodology  
✅ GitHub repository with full code  

**Visual:** Checkmark list with icons

---

### Slide 19: Future Enhancements
**Content:**
- **Model Improvements:**
  - Try larger models (BERT, RoBERTa)
  - Ensemble methods
  - Active learning
- **Deployment:**
  - Docker containerization
  - REST API with FastAPI
  - Cloud deployment (AWS/Azure)
- **Features:**
  - Multi-language support
  - Confidence calibration
  - Explainability (LIME/SHAP)

**Visual:** Roadmap timeline

---

### Slide 20: Conclusion
**Content:**
- **Summary:**
  - Successfully automated news classification
  - Demonstrated CRISP-DM methodology
  - Delivered end-to-end solution
- **Lessons Learned:**
  - Importance of baseline models
  - MLflow for experiment management
  - Balance accuracy vs deployment cost
- **Thank You!**
- **Questions?**

**Visual:** Project logo or final summary graphic

---

## 🎤 Presentation Tips

### Timing Breakdown (10-15 min total)
- Introduction & Problem: 2 min
- Dataset & Methodology: 2 min
- Technical Approach: 3 min
- Results & Demo: 5 min
- Conclusion & Q&A: 3 min

### Speaking Notes
- **Slide 1-3:** Hook audience with business problem
- **Slide 4-7:** Establish credibility with data
- **Slide 8-12:** Technical depth for stakeholders
- **Slide 13-15:** Results speak for themselves
- **Slide 16-17:** Interactive demo engagement
- **Slide 18-20:** Strong closing, future vision

### Presentation Style
- Keep slides visual with minimal text
- Use animations sparingly
- Have backup slides for technical questions
- Practice demo beforehand
- Prepare for Q&A on model choices
