# Emotion Detection System

## Project Overview

Emotion Detection System is a Machine Learning and Natural Language Processing (NLP) project developed using Python and Scikit-learn. The system analyzes textual input and predicts the emotion expressed in the text. It uses feature extraction techniques and multiple supervised machine learning algorithms to classify emotions accurately.

The project supports seven emotion categories:

- Joy
- Fear
- Anger
- Sadness
- Disgust
- Shame
- Guilt

---

## Features

- Text-based emotion classification
- Natural Language Processing (NLP) techniques
- N-gram feature extraction (1-gram to 4-gram)
- Multiple machine learning algorithms for comparison
- Real-time emotion prediction
- Model performance evaluation using training and testing accuracy
- Console-based interactive interface

---

## Technologies Used

### Programming Language
- Python

### Libraries
- Scikit-learn
- Regular Expressions (re)
- Collections (Counter)

### Machine Learning Algorithms
- Support Vector Machine (SVM)
- Linear Support Vector Machine (LinearSVC)
- Random Forest Classifier
- Decision Tree Classifier

---

## Project Architecture

### 1. Data Loading
The dataset is loaded from a text file containing emotion labels and corresponding text samples.

### 2. Text Preprocessing
The input text undergoes preprocessing steps such as:
- Lowercase conversion
- Removal of unwanted characters
- Tokenization

### 3. Feature Extraction
The system generates:
- Unigrams
- Bigrams
- Trigrams
- Four-grams

These features are converted into numerical representations for machine learning models.

### 4. Vectorization
The extracted features are transformed using Scikit-learn's DictVectorizer.

### 5. Model Training
Multiple machine learning models are trained and compared:
- SVM
- Linear SVM
- Random Forest
- Decision Tree

### 6. Performance Evaluation
The models are evaluated using:
- Training Accuracy
- Testing Accuracy

### 7. Real-Time Prediction
Users can enter custom text and receive instant emotion predictions.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/emotion-detection-system.git
cd emotion-detection-system
```

### Install Dependencies

```bash
pip install scikit-learn
```

---

## Dataset Format

The dataset should contain emotion labels and corresponding text entries.

Example:

```text
[1 0 0 0 0 0 0] I am very happy today.
[0 0 1 0 0 0 0] I am extremely angry.
[0 0 0 1 0 0 0] I feel very sad.
```

Emotion Mapping:

| Label | Emotion |
|---------|---------|
| 1 | Joy |
| 2 | Fear |
| 3 | Anger |
| 4 | Sadness |
| 5 | Disgust |
| 6 | Shame |
| 7 | Guilt |

---

## Running the Project

Update the dataset path inside the source code:

```python
file_path = "path/to/your/dataset.txt"
```

Run the program:

```bash
python emotion_detection.py
```

---

## Example Usage

### Input

```text
Enter text: I am excited about my new job.
```

### Output

```text
Predicted Emotion: joy 😂
```

### Input

```text
Enter text: I regret my decision.
```

### Output

```text
Predicted Emotion: guilt 😔
```

---

## Sample Performance Output

```text
| Classifier                | Training Accuracy | Test Accuracy |
|---------------------------|------------------|---------------|
| SVC                       | 0.92            | 0.81          |
| LinearSVC                 | 0.91            | 0.83          |
| RandomForestClassifier    | 0.99            | 0.80          |
| DecisionTreeClassifier    | 1.00            | 0.75          |
```

---

## Project Structure

```text
Emotion-Detection-System/
│
├── emotion_detection.py
├── dataset.txt
├── README.md
│
└── requirements.txt
```

---

## Learning Outcomes

Through this project, the following concepts were explored:

- Natural Language Processing (NLP)
- Text Preprocessing
- Feature Engineering
- N-gram Extraction
- Machine Learning Classification
- Model Evaluation
- Real-Time Prediction Systems
- Python-based ML Development

---

## Future Enhancements

- Deep Learning-based emotion detection using LSTM
- Transformer models such as BERT
- Graphical User Interface (GUI)
- Web application deployment using Flask
- Multi-language emotion classification
- Improved dataset and accuracy optimization

---

## Author

**Vinod Reddy**

Data Science Student | Machine Learning Enthusiast | Aspiring Full Stack Developer

---

## License

This project is developed for educational and academic purposes.
