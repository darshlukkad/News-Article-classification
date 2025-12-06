"""
Inference Module: Load trained models and make predictions
"""

import os
import torch
import joblib
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import Tuple, List, Dict
from src.data_prep import clean_text


class BaselinePredictor:
    """Predictor for baseline TF-IDF + Logistic Regression model"""
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize baseline predictor
        
        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = model_dir
        self.vectorizer = None
        self.classifier = None
        self.label_names = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
        
    def load_model(self):
        """Load the trained baseline model"""
        vectorizer_path = os.path.join(self.model_dir, 'baseline_vectorizer.pkl')
        classifier_path = os.path.join(self.model_dir, 'baseline_classifier.pkl')
        
        if not os.path.exists(vectorizer_path) or not os.path.exists(classifier_path):
            raise FileNotFoundError(
                f"Model files not found in {self.model_dir}. "
                "Please train the baseline model first."
            )
        
        self.vectorizer = joblib.load(vectorizer_path)
        self.classifier = joblib.load(classifier_path)
        
        print("✓ Loaded baseline model")
        
    def predict(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Make prediction on a single text
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (predicted_label, confidence_scores)
        """
        if self.vectorizer is None or self.classifier is None:
            self.load_model()
        
        # Preprocess text
        text_clean = clean_text(text)
        
        # Vectorize
        X = self.vectorizer.transform([text_clean])
        
        # Predict
        pred_label = self.classifier.predict(X)[0]
        pred_proba = self.classifier.predict_proba(X)[0]
        
        # Get label name
        predicted_category = self.label_names[pred_label]
        
        # Create confidence scores dict
        confidence_scores = {
            self.label_names[i]: float(pred_proba[i])
            for i in range(len(pred_proba))
        }
        
        return predicted_category, confidence_scores
    
    def predict_batch(self, texts: List[str]) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Make predictions on multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            Tuple of (predicted_labels, confidence_scores_list)
        """
        if self.vectorizer is None or self.classifier is None:
            self.load_model()
        
        # Preprocess texts
        texts_clean = [clean_text(text) for text in texts]
        
        # Vectorize
        X = self.vectorizer.transform(texts_clean)
        
        # Predict
        pred_labels = self.classifier.predict(X)
        pred_probas = self.classifier.predict_proba(X)
        
        # Convert to category names and confidence scores
        predicted_categories = [self.label_names[label] for label in pred_labels]
        
        confidence_scores_list = [
            {self.label_names[i]: float(proba[i]) for i in range(len(proba))}
            for proba in pred_probas
        ]
        
        return predicted_categories, confidence_scores_list


class TransformerPredictor:
    """Predictor for DistilBERT transformer model"""
    
    def __init__(self, model_dir: str = 'models/transformer'):
        """
        Initialize transformer predictor
        
        Args:
            model_dir: Directory containing saved model
        """
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = 128
        
        self.label_names = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
        
    def load_model(self):
        """Load the trained transformer model"""
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}. "
                "Please train the transformer model first."
            )
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Loaded transformer model on {self.device}")
        
    def predict(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Make prediction on a single text
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (predicted_label, confidence_scores)
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
            pred_label = torch.argmax(probs).item()
        
        # Get label name
        predicted_category = self.label_names[pred_label]
        
        # Create confidence scores dict
        confidence_scores = {
            self.label_names[i]: float(probs[i])
            for i in range(len(probs))
        }
        
        return predicted_category, confidence_scores
    
    def predict_batch(self, texts: List[str], batch_size: int = 16) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Make predictions on multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (predicted_labels, confidence_scores_list)
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        predicted_categories = []
        confidence_scores_list = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                pred_labels = torch.argmax(probs, dim=1)
            
            # Convert to category names and confidence scores
            for j in range(len(batch_texts)):
                pred_label = pred_labels[j].item()
                predicted_categories.append(self.label_names[pred_label])
                
                confidence_scores = {
                    self.label_names[k]: float(probs[j][k])
                    for k in range(len(probs[j]))
                }
                confidence_scores_list.append(confidence_scores)
        
        return predicted_categories, confidence_scores_list


def demo_predictions():
    """Demo function showing how to use the predictors"""
    print("\n" + "="*70)
    print("INFERENCE DEMO")
    print("="*70)
    
    # Sample texts
    sample_texts = [
        "Lakers defeat Celtics in overtime thriller to win NBA championship",
        "Federal Reserve announces interest rate hike to combat inflation",
        "New quantum computer breakthrough could revolutionize cryptography",
        "UN Security Council holds emergency meeting on global crisis"
    ]
    
    print("\nTrying to load baseline model...")
    try:
        baseline_predictor = BaselinePredictor()
        baseline_predictor.load_model()
        
        print("\nBaseline Model Predictions:")
        print("-" * 70)
        
        for text in sample_texts:
            category, scores = baseline_predictor.predict(text)
            print(f"\nText: {text}")
            print(f"Predicted: {category}")
            print(f"Confidence: {scores[category]:.4f}")
            
    except FileNotFoundError as e:
        print(f"✗ {e}")
    
    print("\n\nTrying to load transformer model...")
    try:
        transformer_predictor = TransformerPredictor()
        transformer_predictor.load_model()
        
        print("\nTransformer Model Predictions:")
        print("-" * 70)
        
        for text in sample_texts:
            category, scores = transformer_predictor.predict(text)
            print(f"\nText: {text}")
            print(f"Predicted: {category}")
            print(f"Confidence: {scores[category]:.4f}")
            
    except FileNotFoundError as e:
        print(f"✗ {e}")
    
    print("\n" + "="*70)
    print("To use inference, first train the models:")
    print("  1. python src/train_baseline.py")
    print("  2. python src/train_transformer.py")
    print("="*70)


if __name__ == "__main__":
    demo_predictions()
