# CS280 Intelligent Systems

Classical machine learning coursework from the Master of Science in Computer Science program (2016-2017). I developed several machine learning solutions such as spam detection, neural network, ensemble methods, and unsupervised learning through systematic exploration of classification techniques, from probabilistic models to neural networks to ensemble approaches. I focused on addressing real-world challenges such as handling imbalanced datasets, hyperparameter optimization, and model evaluation across multiple metrics.

## Learning Objectives

Through this course, I gained proficiency in:

- **Classification algorithms**: Naive Bayes probabilistic classifiers, neural networks with backpropagation, ensemble methods combining multiple classifiers, support vector machines
- **Data preprocessing**: Feature extraction, handling imbalanced datasets using SMOTE (Synthetic Minority Over-sampling Technique), train-validation-test splitting
- **Model optimization**: Hyperparameter tuning via grid search, cross-validation for robust evaluation, performance metric analysis (precision, recall, F1-score, accuracy)
- **Unsupervised learning**: Independent Component Analysis for feature extraction
- **Implementation practices**: Vectorized computation with NumPy for performance, code optimization for production use, model evaluation metrics

## Coding Exercises and Problem Sets

Each programming assignment contains problem specification, Python implementation, test datasets and results

**PA1 - Spam Filtering**: Implemented a Naive Bayes classifier for email classification achieving high precision and recall on the TREC corpus. Applied Laplace smoothing and mutual information feature selection to optimize performance.

**PA2 - Neural Networks**: Built multi-layer perceptron from scratch using NumPy with adaptive training parameters. Applied SMOTE to address the accuracy paradox in imbalanced datasets, achieving balanced performance across minority and majority classes.

**PA3 - Ensemble of Classifiers**: Compared multiple ensemble techniques combining diverse base classifiers to improve predictive robustness and generalization.

**PA4 - Independent Component Analysis**: Implemented ICA for feature extraction and dimensionality reduction on complex datasets.

**Final Project**: Developed gender classification pipeline using both support vector machines and artificial neural networks, comparing their performance and tuning kernels and network architectures for optimal results using grid search.

## Software Stack

- Python
- NumPy
- scikit-learn
- LIBSVM
