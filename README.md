# Iris Classification with Gaussian Naive Bayes

This project demonstrates the classification of the Iris dataset using the Gaussian Naive Bayes algorithm. The Iris dataset is loaded from scikit-learn's built-in datasets and used to train and evaluate a classification model.

## Project Overview

- **Dataset**: Iris dataset from `sklearn.datasets`
- **Algorithm**: Gaussian Naive Bayes Classifier
- **Objective**: Classify Iris flowers into one of three species (Setosa, Versicolour, or Virginica) based on their physical attributes.
- **Tools Used**: Python, Pandas, scikit-learn

## Steps:
1. Load the Iris dataset.
2. Preprocess the data and create features and target variables.
3. Split the data into training and testing sets.
4. Train the model using Gaussian Naive Bayes.
5. Evaluate the model using accuracy score, confusion matrix, and classification report.

## Results:
- **Accuracy**: 96.67%
- **Confusion Matrix**:
    - [[x, y], [z, w], [a, b]]
- **Classification Report**:  
    ```
    precision    recall  f1-score   support
    0       x.xx      x.xx      x.xx      xxx
    1       y.yy      y.yy      y.yy      yyy
    2       z.zz      z.zz      z.zz      zzz
    ```

## How to Run

### Prerequisites
- Python 3.x
- Required Libraries (see `requirements.txt`)

### Steps to Run:
1. Clone the repository.
    ```bash
    git clone https://github.com/your-username/iris_classification.git
    ```
2. Navigate to the project directory.
    ```bash
    cd iris_classification
    ```
3. Install the required dependencies.
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter notebook or the Python script:
    - Jupyter Notebook: Open the `iris_classification.ipynb` file and run the cells.
    - Python Script: Execute the Python file in your terminal.
    ```bash
    python iris_classification.py
    ```

## License
This project is licensed under the MIT License.
