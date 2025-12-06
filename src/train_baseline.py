"""
Baseline Model Training: TF-IDF + Logistic Regression
With MLflow tracking
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)
import joblib
import mlflow
import mlflow.sklearn
from datasets import load_dataset

# Import data preparation utilities
from data_prep import clean_text


class BaselineModel:
    """
    Baseline model using TF-IDF and Logistic Regression
    """
    
    def __init__(self, 
                 max_features: int = 5000,
                 ngram_range: tuple = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.95,
                 random_state: int = 42):
        """
        Initialize baseline model
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            random_state: Random seed
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state
        
        # Initialize models
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs',
            multi_class='multinomial',
            n_jobs=-1
        )
        
        self.label_names = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
        
    def load_data(self, cache_dir: str = "./data"):
        """Load AG News dataset"""
        print("Loading AG News dataset...")
        dataset = load_dataset("ag_news", cache_dir=cache_dir)
        
        self.train_texts = dataset['train']['text']
        self.train_labels = dataset['train']['label']
        self.test_texts = dataset['test']['text']
        self.test_labels = dataset['test']['label']
        
        print(f"✓ Loaded {len(self.train_texts)} training samples")
        print(f"✓ Loaded {len(self.test_texts)} test samples")
        
    def preprocess_data(self):
        """Preprocess text data"""
        print("\nPreprocessing text data...")
        
        # Clean training texts
        self.train_texts_clean = [clean_text(text) for text in self.train_texts]
        self.test_texts_clean = [clean_text(text) for text in self.test_texts]
        
        print("✓ Text preprocessing complete")
        
    def train(self):
        """Train the baseline model"""
        print("\nTraining baseline model...")
        print("="*60)
        
        start_time = time.time()
        
        # Fit TF-IDF vectorizer
        print("Fitting TF-IDF vectorizer...")
        X_train = self.vectorizer.fit_transform(self.train_texts_clean)
        print(f"✓ TF-IDF matrix shape: {X_train.shape}")
        
        # Train logistic regression
        print("Training Logistic Regression classifier...")
        self.classifier.fit(X_train, self.train_labels)
        
        training_time = time.time() - start_time
        print(f"✓ Training complete in {training_time:.2f} seconds")
        
        return training_time
        
    def evaluate(self):
        """Evaluate the model"""
        print("\nEvaluating model...")
        print("="*60)
        
        # Transform test data
        X_test = self.vectorizer.transform(self.test_texts_clean)
        
        # Make predictions
        start_time = time.time()
        y_pred = self.classifier.predict(X_test)
        inference_time = (time.time() - start_time) / len(self.test_texts)
        
        # Calculate metrics
        accuracy = accuracy_score(self.test_labels, y_pred)
        f1_macro = f1_score(self.test_labels, y_pred, average='macro')
        f1_micro = f1_score(self.test_labels, y_pred, average='micro')
        precision = precision_score(self.test_labels, y_pred, average='macro')
        recall = recall_score(self.test_labels, y_pred, average='macro')
        
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  F1 (Macro):  {f1_macro:.4f}")
        print(f"  F1 (Micro):  {f1_micro:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"  Avg Inference Time: {inference_time*1000:.2f}ms per sample")
        
        # Classification report
        print("\nClassification Report:")
        target_names = [self.label_names[i] for i in sorted(self.label_names.keys())]
        print(classification_report(self.test_labels, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.test_labels, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'precision': precision,
            'recall': recall,
            'inference_time_ms': inference_time * 1000
        }
        
        return metrics, cm, y_pred
    
    def plot_confusion_matrix(self, cm, save_path: str = 'figures/confusion_matrix_baseline.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_names.values(),
                   yticklabels=self.label_names.values(),
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix - Baseline Model (TF-IDF + LogReg)', fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
        
        plt.show()
        
        return save_path
    
    def save_model(self, model_dir: str = 'models'):
        """Save the trained model"""
        os.makedirs(model_dir, exist_ok=True)
        
        vectorizer_path = os.path.join(model_dir, 'baseline_vectorizer.pkl')
        classifier_path = os.path.join(model_dir, 'baseline_classifier.pkl')
        
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.classifier, classifier_path)
        
        print(f"\n✓ Model saved to {model_dir}/")
        
        return vectorizer_path, classifier_path


def train_with_mlflow():
    """Train baseline model with MLflow tracking"""
    
    # Set MLflow experiment
    mlflow.set_experiment("AG_News_Classification")
    
    # Start MLflow run
    with mlflow.start_run(run_name="baseline_tfidf_logreg"):
        
        print("\n" + "="*60)
        print("TRAINING BASELINE MODEL WITH MLFLOW TRACKING")
        print("="*60)
        
        # Initialize model
        model = BaselineModel(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            random_state=42
        )
        
        # Log parameters
        mlflow.log_param("model_type", "baseline")
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("classifier", "LogisticRegression")
        mlflow.log_param("max_features", model.max_features)
        mlflow.log_param("ngram_range", model.ngram_range)
        mlflow.log_param("min_df", model.min_df)
        mlflow.log_param("max_df", model.max_df)
        
        # Load and preprocess data
        model.load_data()
        model.preprocess_data()
        
        # Train model
        training_time = model.train()
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Evaluate model
        metrics, cm, y_pred = model.evaluate()
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Plot and log confusion matrix
        cm_path = model.plot_confusion_matrix(cm)
        mlflow.log_artifact(cm_path)
        
        # Save and log model
        vectorizer_path, classifier_path = model.save_model()
        mlflow.log_artifact(vectorizer_path)
        mlflow.log_artifact(classifier_path)
        
        # Log model with MLflow
        mlflow.sklearn.log_model(model.classifier, "model")
        
        print("\n" + "="*60)
        print("BASELINE MODEL TRAINING COMPLETE!")
        print("="*60)
        print(f"\n✓ Accuracy: {metrics['accuracy']:.4f}")
        print(f"✓ F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"✓ Training Time: {training_time:.2f}s")
        print(f"\n✓ All metrics logged to MLflow")
        print(f"✓ Run 'mlflow ui' to view results")
        
        return model, metrics


if __name__ == "__main__":
    model, metrics = train_with_mlflow()
