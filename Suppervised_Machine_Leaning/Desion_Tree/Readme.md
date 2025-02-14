# Decision Tree Algorithm

## 1. Introduction to Decision Trees
A Decision Tree is a supervised machine learning algorithm used for both classification and regression tasks. It is a tree-like structure where internal nodes represent decisions (based on features), branches represent outcomes, and leaf nodes represent final predictions.

### Example Use Case
Imagine you are predicting whether a person will buy a car based on income, age, and credit score. A decision tree will split the data at each step based on these attributes until it reaches a conclusion (Yes/No).

## 2. Structure of a Decision Tree
A decision tree consists of:

- **Root Node**: The topmost node representing the entire dataset.
- **Internal Nodes**: Intermediate nodes that test a feature.
- **Branches**: Paths from one node to another.
- **Leaf Nodes**: Final nodes representing a decision (class label or value).

## 3. How a Decision Tree Works
- **Feature Selection** – The algorithm selects the best feature to split the data.
- **Splitting** – Data is divided into subsets based on feature values.
- **Stopping Criteria** – The tree stops splitting when:
  - All instances in a node belong to the same class.
  - The tree reaches a predefined depth.
  - The information gain is below a threshold.
- **Prediction** – A new instance follows the tree structure until it reaches a leaf node.

## 4. Key Concepts in Decision Trees
### A. Splitting Criteria (How Trees Choose the Best Feature)
Decision trees use mathematical measures to choose the best feature for splitting.

#### (i) Entropy and Information Gain (for Classification)
- **Entropy** measures the impurity in a dataset.
- **Information Gain (IG)** is the reduction in entropy after a split.

**Formula for Entropy:**
```math
Entropy(S) = -\sum p_i \log_2(p_i)
```
where \( p_i \) is the probability of class \( i \).

**Formula for Information Gain:**
```math
IG = Entropy(parent) - \sum \left( \frac{samples_{child}}{samples_{parent}} \times Entropy(child) \right)
```
A split with higher Information Gain is preferred.

#### (ii) Gini Index (for Classification)
- Measures the impurity of a dataset like entropy but is computationally faster.

**Formula:**
```math
Gini(S) = 1 - \sum p_i^2
```
Lower Gini Index means purer nodes.

#### (iii) Mean Squared Error (for Regression)
For decision trees used in regression, the **Mean Squared Error (MSE)** is used:
```math
MSE = \frac{1}{n} \sum (y_i - \hat{y})^2
```
where \( y_i \) is the actual value and \( \hat{y} \) is the predicted value.

## 5. Types of Decision Trees
- **Classification Tree** – Used when the output is a **category** (e.g., Spam or Not Spam).
- **Regression Tree** – Used when the output is a **numerical value** (e.g., predicting house prices).

## 6. Advantages & Disadvantages
### ✅ Advantages:
- Easy to understand and interpret.
- Handles both numerical and categorical data.
- Requires little data preprocessing.

### ❌ Disadvantages:
- **Overfitting**: Deep trees can become complex and fit noise instead of patterns.
- **Biased splits**: If a feature has many unique values, it may dominate the tree.
- **Instability**: Small changes in data can create different trees.

**Solution:** Use **pruning** or ensembles like **Random Forest** to improve decision trees.

## 7. Decision Tree Pruning (Avoiding Overfitting)
Pruning reduces the complexity of the tree:

- **Pre-pruning (Early Stopping)** – Stop the tree before it becomes too deep.
- **Post-pruning (Prune After Training)** – Remove weak branches after building the tree.

