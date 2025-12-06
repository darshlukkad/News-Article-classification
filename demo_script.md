# Demo Video Script: News Article Topic Classification

**Duration:** 5-10 minutes  
**Format:** Screen recording with voiceover  
**Tools:** Screen recording software (QuickTime, OBS, Loom)

---

## 🎬 Video Structure

### Section 1: Introduction (30 seconds)
**Visual:** Title screen or GitHub repo homepage

**Script:**
> "Hi everyone! Today I'm excited to show you my end-to-end machine learning project for news article topic classification. In just 48 hours, I built a complete pipeline that automatically categorizes news articles into four topics: World, Sports, Business, and Science & Technology. This project demonstrates the full CRISP-DM methodology, from data understanding to deployment, using the AG News dataset from HuggingFace."

---

### Section 2: Project Overview (45 seconds)
**Visual:** Navigate through GitHub repository structure

**Script:**
> "Let me give you a quick tour of the project structure. We have our data directory for caching the AG News dataset, a notebooks folder with our training pipeline, and the src directory containing all our modular code. The app folder has our Streamlit deployment, and we use MLflow to track all our experiments. Everything is well-organized following software engineering best practices with proper documentation, requirements file, and a gitignore to keep our repo clean."

**Actions:**
- Scroll through folder structure
- Show README briefly
- Highlight key files

---

### Section 3: Dataset Introduction (30 seconds)
**Visual:** Show Jupyter notebook or Python script loading data

**Script:**
> "The AG News dataset is perfect for this task. It contains 120,000 training samples and 7,600 test samples, perfectly balanced across four categories. Each sample includes a news article title and description that we'll use to predict the category."

**Actions:**
```python
from datasets import load_dataset
dataset = load_dataset("ag_news")
print(dataset)
```
- Show dataset loading
- Display sample articles
- Show class distribution

---

### Section 4: Baseline Model (1 minute)
**Visual:** Run baseline training script

**Script:**
> "I started with a baseline model using TF-IDF vectorization and logistic regression. This gives us a fast, interpretable benchmark to compare against. Watch as it trains in just a couple of minutes..."

**Actions:**
```bash
python src/train_baseline.py
```
- Start training script
- Show training progress
- Display final metrics

**Script (continued):**
> "As you can see, we achieved [XX%] accuracy with an F1-score of [X.XX]. Not bad for a simple model! All these metrics are automatically logged to MLflow for tracking."

---

### Section 5: Transformer Model (1.5 minutes)
**Visual:** Run transformer training script

**Script:**
> "Now let's train our transformer model using DistilBERT. DistilBERT is a smaller, faster version of BERT that retains 97% of its performance but trains much quicker. I'm fine-tuning it for three epochs on our AG News dataset..."

**Actions:**
```bash
python src/train_transformer.py
```
- Start training (show accelerated or skip to end)
- Show training progress bars
- Display epoch losses

**Script (continued):**
> "The training takes about 30 minutes, so I've sped this up for the demo. Notice how the loss decreases each epoch, and we're tracking everything in MLflow—parameters, metrics, and model artifacts."

---

### Section 6: MLflow Experiment Tracking (1 minute)
**Visual:** Open MLflow UI

**Script:**
> "Let's check out MLflow to compare our experiments. Here you can see all the runs we've performed, with the baseline and transformer models clearly tracked."

**Actions:**
```bash
mlflow ui
```
- Navigate to localhost:5000
- Show experiment list
- Compare metrics between runs
- Show confusion matrix artifact
- Display model registry

**Script (continued):**
> "MLflow gives us complete visibility into our experiments. We can compare accuracy, F1-scores, and even visualize confusion matrices. This makes it easy to select the best model for deployment."

---

### Section 7: Model Evaluation (1 minute)
**Visual:** Show evaluation results and visualizations

**Script:**
> "Let's look at the evaluation results in detail. Here's the confusion matrix for both models—notice how the transformer significantly reduces misclassifications, especially between Business and Sci/Tech categories."

**Actions:**
- Show confusion matrices side by side
- Display F1-score comparison chart
- Show classification report

**Script (continued):**
> "The transformer model achieved [XX%] accuracy and an F1-score of [X.XX], substantially outperforming our baseline. Most importantly, it maintains high performance across all four categories."

---

### Section 8: Streamlit Application Demo (2 minutes)
**Visual:** Launch and use Streamlit app

**Script:**
> "Now for the exciting part—the deployment! I built a Streamlit web application that loads our best model and provides real-time predictions. Let's launch it..."

**Actions:**
```bash
streamlit run app/streamlit_app.py
```
- Wait for app to load
- Show landing page

**Script (continued):**
> "The interface is simple and intuitive. Let me try a few examples..."

**Test Case 1:**
**Input:** "Lakers defeat Celtics in overtime thriller to win NBA championship"
**Action:** Paste text, click Classify
**Expected:** Sports category

**Script:**
> "As expected, it correctly identifies this as a Sports article with high confidence."

**Test Case 2:**
**Input:** "Federal Reserve announces interest rate hike to combat inflation"
**Action:** Paste text, click Classify
**Expected:** Business category

**Script:**
> "And here we see a Business article classified correctly."

**Test Case 3:**
**Input:** "New quantum computer breakthrough could revolutionize cryptography"
**Action:** Paste text, click Classify
**Expected:** Sci/Tech category

**Script:**
> "Science and Technology—perfect! Notice how it also shows confidence scores for all four categories, giving users full transparency into the model's decision."

**Test Case 4 (Edge Case):**
**Input:** "Apple reports record quarterly earnings driven by iPhone sales"
**Action:** Paste text, click Classify

**Script:**
> "This one's interesting—it could be Business or Sci/Tech since it's about a tech company's finances. Let's see... [wait for result]. The model classified it as [result], which makes sense given the emphasis on earnings and sales."

---

### Section 9: Code Walkthrough (1 minute)
**Visual:** Show key code snippets

**Script:**
> "Let me quickly show you some key code. Here's how we load and preprocess the data, here's the transformer training loop with MLflow logging, and here's the inference function used by the Streamlit app."

**Actions:**
- Open `src/data_prep.py` - show data loading
- Open `src/train_transformer.py` - show training loop
- Open `app/streamlit_app.py` - show prediction function

**Script (continued):**
> "Everything is modular and reusable. You could easily adapt this code for other text classification tasks or different datasets."

---

### Section 10: Documentation (30 seconds)
**Visual:** Browse documentation files

**Script:**
> "Documentation is crucial for any project. I've created comprehensive docs including a README with setup instructions, a complete CRISP-DM methodology document covering all six phases, a presentation slides outline, and even this demo script!"

**Actions:**
- Scroll through README.md
- Show crisp_dm.md
- Show slides_outline.md

---

### Section 11: Conclusion (30 seconds)
**Visual:** Return to GitHub repo or project summary

**Script:**
> "So there you have it—a complete end-to-end machine learning project built in 48 hours. We've got data loading from HuggingFace, baseline and transformer models, MLflow experiment tracking, comprehensive evaluation, and a deployed Streamlit application. All the code is available on GitHub, and I've included detailed documentation to help anyone reproduce or build upon this work."

**Script (continued):**
> "Some future enhancements could include trying larger transformer models, adding a REST API for programmatic access, or deploying to the cloud with Docker. Thanks for watching, and feel free to reach out if you have any questions!"

---

## 🎥 Recording Tips

### Before Recording:
- [ ] Close unnecessary applications
- [ ] Clear terminal history
- [ ] Prepare sample texts for demo
- [ ] Have all windows ready
- [ ] Test audio levels
- [ ] Check screen resolution (1920x1080 recommended)

### During Recording:
- [ ] Speak clearly and at a steady pace
- [ ] Pause between sections
- [ ] Show only relevant terminal output
- [ ] Highlight important parts with cursor
- [ ] Keep mouse movements smooth
- [ ] Have water nearby

### After Recording:
- [ ] Edit out long pauses
- [ ] Add title cards between sections
- [ ] Include annotations for clarity
- [ ] Add background music (optional, low volume)
- [ ] Export at 1080p
- [ ] Upload to YouTube/LinkedIn

---

## 📝 Backup Script (If Something Fails)

**For Training Failures:**
> "If you encounter any issues during training, make sure you have all dependencies installed from requirements.txt and that you have sufficient disk space for the dataset cache."

**For Streamlit Issues:**
> "If the Streamlit app doesn't load, ensure the model file path is correct and that MLflow has saved the model artifacts properly."

**For Demo Artifacts:**
> "All visualizations and evaluation results are saved in the figures directory and can be viewed separately if needed."

---

## 🎬 Alternative: Live Demo Script

If doing a live demo instead of recording:

1. **Have backup saved outputs** in case training takes too long
2. **Pre-load models** to avoid waiting
3. **Prepare 5-6 test articles** across all categories
4. **Have MLflow UI open** in a separate tab
5. **Rehearse transitions** between sections
6. **Time yourself** to stay within limits

---

## 📹 Video Checklist

- [ ] Introduction covers project goal
- [ ] Dataset is clearly explained
- [ ] Both models are demonstrated
- [ ] MLflow tracking is shown
- [ ] Evaluation metrics are displayed
- [ ] Streamlit demo works smoothly
- [ ] Multiple test cases shown
- [ ] Code quality is highlighted
- [ ] Documentation is mentioned
- [ ] Future work is discussed
- [ ] Clear call-to-action at end
- [ ] Video length is 5-10 minutes
- [ ] Audio is clear throughout
- [ ] Visuals are easy to follow
