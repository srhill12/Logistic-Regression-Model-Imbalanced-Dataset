```markdown
# Logistic Regression Model on Imbalanced Dataset

This project demonstrates the application of a Logistic Regression model on an imbalanced dataset. The dataset includes permission data for various applications, and the target variable (`Result`) is imbalanced, with significantly more negative results (`0`) than positive ones (`1`).

## Dataset

The dataset used for this project is loaded directly from a URL:

```python
df = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m14/lesson_1/datasets/app-data-imbalanced.csv")
df.head()
```

The dataset includes various permissions required by applications and a `Result` column indicating the outcome (0 or 1). Here is an overview of the data:

| Permission 1                         | Permission 2                                | ... | Permission N                         | Result |
|--------------------------------------|---------------------------------------------|-----|--------------------------------------|--------|
| android.permission.GET_ACCOUNTS      | com.sonyericsson.home.permission.BROADCAST_BADGE | ... | android.permission.RECEIVE_SMS      | 0      |
| android.permission.GET_ACCOUNTS      | com.sonyericsson.home.permission.BROADCAST_BADGE | ... | android.permission.RECEIVE_SMS      | 0      |
| ...                                  | ...                                         | ... | ...                                  | ...    |

### Imbalance in Data

The target variable is highly imbalanced:

- Negative (`0`): 14,632
- Positive (`1`): 2,000

```python
df['Result'].value_counts()
```

## Model Training

### Data Preparation

I split the data into features (`X`) and the target variable (`y`):

```python
X = df.drop(columns=['Result'])
y = df['Result']
```

### Model Creation and Training

A Logistic Regression model is used to classify the data:

```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X, y)
```

### Model Performance

#### Accuracy

The model achieved an accuracy of approximately 89.24%:

```python
classifier.score(X, y)
```

#### Confusion Matrix

The confusion matrix shows the distribution of true positives, false positives, true negatives, and false negatives:

```python
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y, classifier.predict(X), labels=[1, 0]))
```

|              | Predicted Positive (1) | Predicted Negative (0) |
|--------------|------------------------|------------------------|
| Actual Positive (1) | 322                    | 1678                   |
| Actual Negative (0) | 111                    | 14521                  |

#### Classification Report

The classification report provides precision, recall, f1-score, and support for each class:

```python
from sklearn.metrics import classification_report

print(classification_report(y, classifier.predict(X), labels=[1, 0]))
```

- **Precision**: 0.74 (for class 1)
- **Recall**: 0.16 (for class 1)
- **F1-Score**: 0.26 (for class 1)

#### Balanced Accuracy Score

The balanced accuracy score, which accounts for class imbalance, is calculated as follows:

```python
from sklearn.metrics import balanced_accuracy_score

print(balanced_accuracy_score(y, classifier.predict(X)))
```

- **Balanced Accuracy Score**: 0.577

#### ROC AUC Score

The ROC AUC score is computed to evaluate the model's ability to distinguish between classes:

```python
from sklearn.metrics import roc_auc_score

pred_probas = classifier.predict_proba(X)
pred_probas_firsts = [prob[1] for prob in pred_probas]
print(roc_auc_score(y, pred_probas_firsts))
```

- **ROC AUC Score**: 0.714

## Conclusion

The Logistic Regression model achieved a reasonable accuracy of 89.24%. However, the model's performance on the minority class (`1`) indicates significant challenges, with a low recall and balanced accuracy score. The ROC AUC score suggests that while the model has some predictive power, it struggles with the imbalanced nature of the dataset.

## Future Work

- **Resampling Techniques**: To improve model performance, I may consider applying resampling techniques like SMOTE or undersampling.
- **Model Tuning**: Further tuning of hyperparameters or trying different algorithms (e.g., Random Forest) could improve results.
- **Feature Engineering**: Investigating and engineering more meaningful features may enhance the model's ability to capture the underlying patterns in the data.
```
