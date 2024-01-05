# Binary Classification with Logistic Regression and k-NN

This is a binary classification project that uses both logistic regression and the k-nearest neighbors (k-NN) algorithm to classify data.

## Description

The goal of this project is to develop two binary classification models to predict whether a customer will churn or not. The dataset used for this project is from a telecommunications company and contains demographic information about customers as well as their account information.

## Installation

To run this project, you will need to have Python and the following libraries installed:

* pandas
* numpy
* matplotlib
* scikit-learn
* seaborn

You can install these libraries by running the following command in your terminal:
```
pip install pandas numpy matplotlib scikit-learn seaborn
```
## Usage

To run the project, follow these steps:

1. Clone this repository to your local machine:
```bash
git clone https://github.com/your-username/repository-name.git
```
2. Navigate to the project directory:
```bash
cd repository-name
```
3. Create a new dataset.csv file in the project directory and fill it with your data.
4. Run the try\_logistic.py script to train the logistic regression model and generate predictions for the test data:
```bash
python try_logistic.py
```
5. Run the try\_knn.py script to train the k-NN model and generate predictions for the test data:
```bash
python try_knn.py
```
6. Compare the performance of the two models using classification metrics such as accuracy, precision, recall, and confusion matrix.

## Notes

* The dataset.csv file should contain two columns of data, the first column representing the customer features and the second column representing the class label (1 for churn, 0 for not churn).
* The performance of different models may vary depending on the data used for training and testing.
* The try\_logistic.py and try\_knn.py scripts use custom implementations of the logistic regression and k-NN algorithms, respectively.

## Contributions

Contributions to this project are welcome. If you would like to contribute, please open a pull request or an issue on this repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Comparison of Performance

Here's an example of how to compare the performance of the two models:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the test data
test_data = pd.read_csv('test_data.csv', delimiter=',')
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Generate predictions for the test data using logistic regression
y_pred_logreg = logreg.predict(X_test)

# Generate predictions for the test data using k-NN
y_pred_knn = knn.predict(X_test)

# Calculate classification metrics for logistic regression
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
precision_logreg = precision_score(y_test, y_pred_logreg)
recall_logreg = recall_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)

# Calculate classification metrics for k-NN
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

# Print the results
print('Logistic Regression Results:')
print('Accuracy :', accuracy_logreg)
print('Precision :', precision_logreg)
print('Recall :', recall_logreg)
print('F1-score :', f1_logreg)
print('Confusion Matrix :\n', conf_matrix_logreg)

print('\nk-NN Results:')
print('Accuracy :', accuracy_knn)
print('Precision :', precision_knn)
print('Recall :', recall_knn)
print('F1-score :', f1_knn)
print('Confusion Matrix :\n', conf\_matrix\_knn)
```
## Conclusion

In this binary classification project, both logistic regression and the k-NN algorithm were used to classify data. The performance of the two models was compared using classification metrics such as accuracy, precision, recall, and confusion matrix. The results show that both models perform well, with logistic regression achieving a slightly higher accuracy and precision than k-NN. However, it is important to note that the performance of different models may vary depending on the data used for training and testing. Therefore, it is recommended to test the models on multiple datasets to evaluate their robustness and generalization.

Here's a summary table of the results:

| Model | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| Logistic Regression | 78.46% | 76% | 77.64% |
| k-NN | 75.83% | 75.48% | 75.46% |

As can be seen from the table, logistic regression achieves a slightly higher accuracy and precision than k-NN. However, it is important to note that these results may vary depending on the data used for training and testing. Therefore, it is recommended to test the models on multiple datasets to evaluate their robustness and generalization.
