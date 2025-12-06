"""
Data Preparation Module for AG News Classification
Loads AG News dataset from HuggingFace and performs EDA
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from collections import Counter
import re
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class AGNewsDataPrep:
    """
    Data preparation and EDA class for AG News dataset
    """
    
    def __init__(self, cache_dir: str = "./data"):
        """
        Initialize data preparation
        
        Args:
            cache_dir: Directory to cache dataset
        """
        self.cache_dir = cache_dir
        self.dataset = None
        self.train_df = None
        self.test_df = None
        self.label_names = {
            0: "World",
            1: "Sports", 
            2: "Business",
            3: "Sci/Tech"
        }
        
    def load_data(self) -> None:
        """Load AG News dataset from HuggingFace"""
        print("Loading AG News dataset from HuggingFace...")
        self.dataset = load_dataset("ag_news", cache_dir=self.cache_dir)
        
        # Convert to pandas for easier manipulation
        self.train_df = pd.DataFrame(self.dataset['train'])
        self.test_df = pd.DataFrame(self.dataset['test'])
        
        # Add label names
        self.train_df['label_name'] = self.train_df['label'].map(self.label_names)
        self.test_df['label_name'] = self.test_df['label'].map(self.label_names)
        
        print(f"✓ Loaded {len(self.train_df)} training samples")
        print(f"✓ Loaded {len(self.test_df)} test samples")
        
    def get_basic_stats(self) -> Dict:
        """Get basic dataset statistics"""
        stats = {
            'train_size': len(self.train_df),
            'test_size': len(self.test_df),
            'num_classes': len(self.label_names),
            'classes': self.label_names
        }
        return stats
    
    def analyze_text_lengths(self, save_fig: bool = True) -> None:
        """Analyze and plot text length distributions"""
        print("\nAnalyzing text lengths...")
        
        # Calculate text lengths
        self.train_df['text_length'] = self.train_df['text'].str.len()
        self.test_df['text_length'] = self.test_df['text'].str.len()
        
        self.train_df['word_count'] = self.train_df['text'].str.split().str.len()
        self.test_df['word_count'] = self.test_df['text'].str.split().str.len()
        
        # Print statistics
        print(f"Average character length (train): {self.train_df['text_length'].mean():.1f}")
        print(f"Average word count (train): {self.train_df['word_count'].mean():.1f}")
        
        # Create visualizations
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Character length distribution
        axes[0].hist(self.train_df['text_length'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Character Length')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Text Length (Characters)')
        axes[0].axvline(self.train_df['text_length'].mean(), color='red', 
                       linestyle='--', label=f"Mean: {self.train_df['text_length'].mean():.0f}")
        axes[0].legend()
        
        # Word count distribution
        axes[1].hist(self.train_df['word_count'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Word Count')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Word Count')
        axes[1].axvline(self.train_df['word_count'].mean(), color='red', 
                       linestyle='--', label=f"Mean: {self.train_df['word_count'].mean():.0f}")
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_fig:
            os.makedirs('figures', exist_ok=True)
            plt.savefig('figures/text_length_distribution.png', dpi=300, bbox_inches='tight')
            print("✓ Saved text length distribution plot")
        
        plt.show()
        
    def analyze_class_distribution(self, save_fig: bool = True) -> None:
        """Analyze and plot class distribution"""
        print("\nAnalyzing class distribution...")
        
        # Count samples per class
        train_counts = self.train_df['label_name'].value_counts()
        test_counts = self.test_df['label_name'].value_counts()
        
        print("\nTraining set distribution:")
        print(train_counts)
        print("\nTest set distribution:")
        print(test_counts)
        
        # Create visualizations
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training set distribution
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        train_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black', alpha=0.8)
        axes[0].set_xlabel('Category')
        axes[0].set_ylabel('Number of Samples')
        axes[0].set_title('Training Set: Class Distribution')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, v in enumerate(train_counts):
            axes[0].text(i, v + 500, str(v), ha='center', fontweight='bold')
        
        # Test set distribution
        test_counts.plot(kind='bar', ax=axes[1], color=colors, edgecolor='black', alpha=0.8)
        axes[1].set_xlabel('Category')
        axes[1].set_ylabel('Number of Samples')
        axes[1].set_title('Test Set: Class Distribution')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, v in enumerate(test_counts):
            axes[1].text(i, v + 50, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_fig:
            os.makedirs('figures', exist_ok=True)
            plt.savefig('figures/class_distribution.png', dpi=300, bbox_inches='tight')
            print("✓ Saved class distribution plot")
        
        plt.show()
        
    def show_sample_articles(self, n_samples: int = 2) -> None:
        """Display sample articles from each category"""
        print("\n" + "="*80)
        print("SAMPLE ARTICLES FROM EACH CATEGORY")
        print("="*80)
        
        for label, label_name in self.label_names.items():
            print(f"\n{'='*80}")
            print(f"Category: {label_name} (Label: {label})")
            print(f"{'='*80}")
            
            samples = self.train_df[self.train_df['label'] == label].head(n_samples)
            
            for idx, (i, row) in enumerate(samples.iterrows(), 1):
                print(f"\nExample {idx}:")
                print(f"Text: {row['text'][:300]}...")
                print(f"Length: {len(row['text'])} chars, {len(row['text'].split())} words")
    
    def get_vocabulary_stats(self) -> Dict:
        """Analyze vocabulary statistics"""
        print("\nAnalyzing vocabulary...")
        
        # Combine all text
        all_text = ' '.join(self.train_df['text'].values)
        
        # Tokenize (simple word splitting)
        words = re.findall(r'\b\w+\b', all_text.lower())
        
        # Get statistics
        vocab_size = len(set(words))
        total_words = len(words)
        word_freq = Counter(words)
        most_common = word_freq.most_common(20)
        
        print(f"Total words: {total_words:,}")
        print(f"Unique words (vocabulary): {vocab_size:,}")
        print(f"Average word frequency: {total_words/vocab_size:.2f}")
        
        print("\nMost common words:")
        for word, count in most_common[:10]:
            print(f"  {word}: {count:,}")
        
        return {
            'vocab_size': vocab_size,
            'total_words': total_words,
            'most_common_words': most_common
        }
    
    def plot_word_frequency(self, top_n: int = 20, save_fig: bool = True) -> None:
        """Plot most frequent words"""
        print(f"\nPlotting top {top_n} most frequent words...")
        
        # Get word frequencies
        all_text = ' '.join(self.train_df['text'].values)
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = Counter(words)
        most_common = word_freq.most_common(top_n)
        
        # Create DataFrame for plotting
        words_df = pd.DataFrame(most_common, columns=['word', 'frequency'])
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.barh(words_df['word'], words_df['frequency'], color='steelblue', edgecolor='black', alpha=0.8)
        plt.xlabel('Frequency')
        plt.ylabel('Word')
        plt.title(f'Top {top_n} Most Frequent Words in AG News Dataset')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            os.makedirs('figures', exist_ok=True)
            plt.savefig('figures/word_frequency.png', dpi=300, bbox_inches='tight')
            print("✓ Saved word frequency plot")
        
        plt.show()
    
    def analyze_text_by_category(self, save_fig: bool = True) -> None:
        """Analyze text characteristics by category"""
        print("\nAnalyzing text characteristics by category...")
        
        # Calculate statistics by category
        category_stats = self.train_df.groupby('label_name').agg({
            'text_length': ['mean', 'std'],
            'word_count': ['mean', 'std']
        }).round(2)
        
        print("\nText statistics by category:")
        print(category_stats)
        
        # Create box plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Character length by category
        self.train_df.boxplot(column='text_length', by='label_name', ax=axes[0])
        axes[0].set_xlabel('Category')
        axes[0].set_ylabel('Character Length')
        axes[0].set_title('Text Length Distribution by Category')
        axes[0].get_figure().suptitle('')
        
        # Word count by category
        self.train_df.boxplot(column='word_count', by='label_name', ax=axes[1])
        axes[1].set_xlabel('Category')
        axes[1].set_ylabel('Word Count')
        axes[1].set_title('Word Count Distribution by Category')
        axes[1].get_figure().suptitle('')
        
        plt.tight_layout()
        
        if save_fig:
            os.makedirs('figures', exist_ok=True)
            plt.savefig('figures/text_by_category.png', dpi=300, bbox_inches='tight')
            print("✓ Saved text by category plot")
        
        plt.show()
    
    def get_preprocessed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return preprocessed train and test dataframes
        
        Returns:
            Tuple of (train_df, test_df)
        """
        return self.train_df, self.test_df
    
    def run_full_eda(self) -> None:
        """Run complete exploratory data analysis"""
        print("\n" + "="*80)
        print("STARTING FULL EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Basic stats
        print("\n" + "-"*80)
        print("BASIC STATISTICS")
        print("-"*80)
        stats = self.get_basic_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Class distribution
        self.analyze_class_distribution()
        
        # Text lengths
        self.analyze_text_lengths()
        
        # Text by category
        self.analyze_text_by_category()
        
        # Vocabulary
        self.get_vocabulary_stats()
        
        # Word frequency
        self.plot_word_frequency()
        
        # Sample articles
        self.show_sample_articles()
        
        print("\n" + "="*80)
        print("EDA COMPLETE!")
        print("="*80)


def clean_text(text: str) -> str:
    """
    Basic text cleaning function
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters (keep letters, numbers, spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def main():
    """Main function to run EDA"""
    # Initialize data preparation
    data_prep = AGNewsDataPrep(cache_dir="./data")
    
    # Run full EDA
    data_prep.run_full_eda()
    
    # Save processed data for later use
    train_df, test_df = data_prep.get_preprocessed_data()
    
    print(f"\nData ready for modeling!")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return data_prep


if __name__ == "__main__":
    data_prep = main()
