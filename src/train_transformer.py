"""
Transformer Model Training: DistilBERT for AG News Classification
With MLflow tracking
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from datasets import load_dataset


class AGNewsDataset(Dataset):
    """PyTorch Dataset for AG News"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TransformerModel:
    """
    DistilBERT model for text classification
    """
    
    def __init__(self,
                 model_name: str = 'distilbert-base-uncased',
                 num_labels: int = 4,
                 max_length: int = 128,
                 batch_size: int = 16,
                 learning_rate: float = 2e-5,
                 num_epochs: int = 3,
                 random_state: int = 42):
        """
        Initialize transformer model
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of classification labels
            max_length: Maximum sequence length
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            random_state: Random seed
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.random_state = random_state
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        
        self.label_names = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def load_data(self, cache_dir: str = "./data", sample_fraction: float = 1.0):
        """Load AG News dataset
        
        Args:
            cache_dir: Directory to cache dataset
            sample_fraction: Fraction of training data to use (0.0 to 1.0)
        """
        print("Loading AG News dataset...")
        dataset = load_dataset("ag_news", cache_dir=cache_dir)
        
        self.train_texts = dataset['train']['text']
        self.train_labels = dataset['train']['label']
        self.test_texts = dataset['test']['text']
        self.test_labels = dataset['test']['label']
        
        # Sample training data if fraction < 1.0
        if sample_fraction < 1.0:
            sample_size = int(len(self.train_texts) * sample_fraction)
            np.random.seed(self.random_state)
            indices = np.random.choice(len(self.train_texts), sample_size, replace=False)
            self.train_texts = [self.train_texts[i] for i in indices]
            self.train_labels = [self.train_labels[i] for i in indices]
            print(f"✓ Sampled {sample_fraction*100:.0f}% of training data")
            
            # Also sample test set for faster evaluation
            test_sample_size = int(len(self.test_texts) * sample_fraction * 10)  # 10x training fraction
            test_indices = np.random.choice(len(self.test_texts), min(test_sample_size, len(self.test_texts)), replace=False)
            self.test_texts = [self.test_texts[i] for i in test_indices]
            self.test_labels = [self.test_labels[i] for i in test_indices]
        
        # Create validation split (10% of training data)
        split_idx = int(0.9 * len(self.train_texts))
        self.val_texts = self.train_texts[split_idx:]
        self.val_labels = self.train_labels[split_idx:]
        self.train_texts = self.train_texts[:split_idx]
        self.train_labels = self.train_labels[:split_idx]
        
        print(f"✓ Loaded {len(self.train_texts)} training samples")
        print(f"✓ Loaded {len(self.val_texts)} validation samples")
        print(f"✓ Loaded {len(self.test_texts)} test samples")
        
    def create_dataloaders(self):
        """Create PyTorch DataLoaders"""
        print("\nCreating dataloaders...")
        
        # Create datasets
        train_dataset = AGNewsDataset(
            self.train_texts, self.train_labels, 
            self.tokenizer, self.max_length
        )
        val_dataset = AGNewsDataset(
            self.val_texts, self.val_labels,
            self.tokenizer, self.max_length
        )
        test_dataset = AGNewsDataset(
            self.test_texts, self.test_labels,
            self.tokenizer, self.max_length
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size
        )
        
        print(f"✓ Created dataloaders (batch_size={self.batch_size})")
        
    def setup_training(self):
        """Setup optimizer and scheduler"""
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        print("✓ Setup optimizer and scheduler")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def train(self):
        """Train the transformer model"""
        print("\n" + "="*60)
        print("TRAINING TRANSFORMER MODEL")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_accuracy = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            
            # Log to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        
        training_time = time.time() - start_time
        print(f"\n✓ Training complete in {training_time:.2f} seconds")
        
        return training_time
    
    def evaluate(self):
        """Evaluate on test set"""
        print("\n" + "="*60)
        print("EVALUATING ON TEST SET")
        print("="*60)
        
        self.model.eval()
        predictions = []
        true_labels = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        inference_time = (time.time() - start_time) / len(self.test_texts)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_micro = f1_score(true_labels, predictions, average='micro')
        precision = precision_score(true_labels, predictions, average='macro')
        recall = recall_score(true_labels, predictions, average='macro')
        
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
        print(classification_report(true_labels, predictions, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'precision': precision,
            'recall': recall,
            'inference_time_ms': inference_time * 1000
        }
        
        return metrics, cm, predictions
    
    def plot_training_curves(self, save_path: str = 'figures/training_curves.png'):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        axes[0].plot(epochs, self.train_losses, 'b-', label='Training Loss', marker='o')
        axes[0].plot(epochs, self.val_losses, 'r-', label='Validation Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy curve
        axes[1].plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy', marker='o')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
        
        plt.close()  # Close instead of show to avoid blocking
        
        return save_path
    
    def plot_confusion_matrix(self, cm, save_path: str = 'figures/confusion_matrix_transformer.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=self.label_names.values(),
                   yticklabels=self.label_names.values(),
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix - Transformer Model (DistilBERT)', fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
        
        plt.close()  # Close instead of show to avoid blocking
        
        plt.show()
        
        return save_path
    
    def save_model(self, model_dir: str = 'models/transformer'):
        """Save the trained model"""
        os.makedirs(model_dir, exist_ok=True)
        
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        print(f"\n✓ Model saved to {model_dir}/")
        
        return model_dir


def train_with_mlflow(quick_mode: bool = False):
    """Train transformer model with MLflow tracking
    
    Args:
        quick_mode: If True, use 10% data and 1 epoch for fast training
    """
    
    # Set MLflow experiment
    mlflow.set_experiment("AG_News_Classification")
    
    # Quick mode settings
    num_epochs = 1 if quick_mode else 3
    sample_fraction = 0.005 if quick_mode else 1.0  # 0.5% for quick mode (~675 samples)
    run_name = "transformer_distilbert_quick" if quick_mode else "transformer_distilbert"
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        
        print("\n" + "="*60)
        print("TRAINING TRANSFORMER MODEL WITH MLFLOW TRACKING")
        if quick_mode:
            print("⚡ QUICK MODE: ~675 samples, 1 epoch")
        print("="*60)
        
        # Initialize model
        model = TransformerModel(
            model_name='distilbert-base-uncased',
            num_labels=4,
            max_length=128,
            batch_size=16,
            learning_rate=2e-5,
            num_epochs=num_epochs,
            random_state=42
        )
        
        # Log parameters
        mlflow.log_param("model_type", "transformer")
        mlflow.log_param("model_name", model.model_name)
        mlflow.log_param("max_length", model.max_length)
        mlflow.log_param("batch_size", model.batch_size)
        mlflow.log_param("learning_rate", model.learning_rate)
        mlflow.log_param("num_epochs", model.num_epochs)
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("scheduler", "linear_warmup")
        mlflow.log_param("quick_mode", quick_mode)
        mlflow.log_param("sample_fraction", sample_fraction)
        
        # Load data and create dataloaders
        model.load_data(sample_fraction=sample_fraction)
        model.create_dataloaders()
        
        # Setup training
        model.setup_training()
        
        # Train model
        training_time = model.train()
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Plot training curves
        curves_path = model.plot_training_curves()
        mlflow.log_artifact(curves_path)
        
        # Evaluate model
        metrics, cm, predictions = model.evaluate()
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Plot and log confusion matrix
        cm_path = model.plot_confusion_matrix(cm)
        mlflow.log_artifact(cm_path)
        
        # Save and log model
        model_dir = model.save_model()
        mlflow.log_artifacts(model_dir)
        
        # Log model with MLflow
        mlflow.pytorch.log_model(model.model, "model")
        
        print("\n" + "="*60)
        print("TRANSFORMER MODEL TRAINING COMPLETE!")
        print("="*60)
        print(f"\n✓ Accuracy: {metrics['accuracy']:.4f}")
        print(f"✓ F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"✓ Training Time: {training_time:.2f}s")
        print(f"\n✓ All metrics logged to MLflow")
        print(f"✓ Run 'mlflow ui' to view results")
        
        return model, metrics


if __name__ == "__main__":
    # Use quick_mode=True for fast training (~2 min)
    # Use quick_mode=False for full training (~30 min)
    model, metrics = train_with_mlflow(quick_mode=False)
