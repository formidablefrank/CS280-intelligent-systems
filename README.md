# CS280 - Intelligent Systems

Classical machine learning coursework from the Master of Science in Computer Science program (2016-2017). This repository contains my implementations and analyses of core machine learning algorithms, demonstrating practical experience in supervised learning, classification, and data preprocessing techniques.

## Description

I developed a comprehensive portfolio of machine learning solutions spanning spam detection, neural network design, ensemble methods, and unsupervised learning. Each assignment progressed through systematic exploration of classification techniques, from probabilistic models to neural networks to ensemble approaches. I focused on addressing real-world challenges such as handling imbalanced datasets, hyperparameter optimization, and model evaluation across multiple metrics.

## Technical Skills and Learning Objectives

Through this coursework, I gained proficiency in:

- **Classification algorithms**: Naive Bayes probabilistic classifiers, artificial neural networks with backpropagation, ensemble methods combining multiple classifiers, support vector machines
- **Data preprocessing**: Feature extraction, handling imbalanced datasets using SMOTE (Synthetic Minority Over-sampling Technique), train-validation-test splitting
- **Model optimization**: Hyperparameter tuning via grid search, cross-validation for robust evaluation, performance metric analysis (precision, recall, F1-score, accuracy)
- **Unsupervised learning**: Independent Component Analysis for feature extraction
- **Implementation practices**: Vectorized computation with NumPy for performance, code optimization for production use, systematic evaluation methodologies

## Coding Exercises and Problem Sets

**PA1 - Spam Filtering**: Implemented a Naive Bayes classifier for email classification achieving high precision and recall on the TREC corpus. Applied Laplace smoothing and mutual information feature selection to optimize performance.

**PA2 - Neural Networks**: Built multi-layer perceptron from scratch using NumPy with adaptive training parameters. Applied SMOTE to address the accuracy paradox in imbalanced datasets, achieving balanced performance across minority and majority classes.

**PA3 - Ensemble of Classifiers**: Compared multiple ensemble techniques combining diverse base classifiers to improve predictive robustness and generalization.

**PA4 - Independent Component Analysis**: Implemented ICA for feature extraction and dimensionality reduction on complex datasets.

**Final Project**: Developed gender classification pipeline using both support vector machines and artificial neural networks, comparing their performance and tuning kernels and network architectures for optimal results using grid search.

## Usage

Each programming assignment directory contains:
- Problem specification in the assignment PDFs
- Python implementation
- Test datasets and results

To run an assignment:
```bash
cd "PA[n] - [Assignment Name]"
python [filename].py
```

The project directory contains the final capstone work with documentation and results.

## Software Stack

- Python
- NumPy
- scikit-learn
- LIBSVM

## References

Key concepts implemented include:

- Naive Bayes classification and smoothing techniques
- Feedforward neural networks and backpropagation
- Support vector machines and kernel methods
- Ensemble learning approaches
- SMOTE for handling imbalanced data
- Independent Component Analysis for unsupervised learning
