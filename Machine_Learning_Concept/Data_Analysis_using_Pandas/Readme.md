# Detailed Explanation of `df.describe()` Statistics

When you use `df.describe()`, it generates key statistical metrics for numerical columns. Let's break down each of these metrics in detail with formulas and interpretations.

## 1. `count` – Number of Non-Null Values
The `count` represents how many non-null (non-missing) values are present in a column.

**Formula:**
```
count = Total number of non-null entries in the column
```
For example, if a column has 100 values but 5 are missing (`NaN`), then `count` will be 95.

---

## 2. `mean` – Average of the Values
The `mean` is the arithmetic average of the values in a column.

**Formula:**
```
Mean = (Sum of all values) / (Total count)
```
**Example:**
For values `[10, 20, 30, 40, 50]`:
```
Mean = (10 + 20 + 30 + 40 + 50) / 5 = 30
```
### Interpretation:
- It gives the central value of the dataset.
- Sensitive to extreme values (outliers), meaning a few very high or very low values can significantly impact it.

---

## 3. `std` – Standard Deviation (Measure of Spread)
The **standard deviation (std)** measures how much the values deviate from the mean.

**Formula:**
```
Standard Deviation (σ) = sqrt( Σ (Xi - μ)^2 / (N - 1) )
```
Where:
- `Xi` are individual values.
- `μ` is the mean.
- `N` is the count.

**Example:**
For values `[10, 20, 30, 40, 50]`, the mean is **30**.
```
Std = sqrt((10-30)^2 + (20-30)^2 + (30-30)^2 + (40-30)^2 + (50-30)^2) / 4
    = sqrt(400 + 100 + 0 + 100 + 400) / 4
    = sqrt(250) ≈ 15.81
```
### Interpretation:
- If `std` is **high**, the data points are more spread out.
- If `std` is **low**, the data points are closer to the mean.
- Helps understand the variability in data.

---

## 4. `min` – Minimum Value
The `min` represents the **smallest value** in the column.

**Example:**
For `[10, 20, 30, 40, 50]`, the **min** is `10`.

### Interpretation:
- Useful for finding the lower bound of data.
- If the `min` is very low compared to other values, it might be an outlier.

---

## 5. `25%` – First Quartile (Q1)
The **25th percentile (Q1)** is the value **below which 25% of the data falls**.

### **How It's Calculated:**
- Sort the values in ascending order.
- Find the **25% position** in the sorted list.
- If it's between two values, take the average.

**Example:**
For `[10, 20, 30, 40, 50]`, Q1 is **20**.

### **Interpretation:**
- Helps understand the **lower bound** of the middle 50% of the data.
- If Q1 is very low, it indicates a **skewed** distribution.

---

## 6. `50%` – Median (Q2)
The **50th percentile (Q2)**, or **median**, is the **middle value** of a sorted dataset.

**Example:**
For `[10, 20, 30, 40, 50]`, the **median (Q2)** is `30`.

### **Interpretation:**
- The median is a **better measure of central tendency** than the mean when there are **outliers**.
- Unlike the mean, the median **isn't affected by extreme values**.

---

## 7. `75%` – Third Quartile (Q3)
The **75th percentile (Q3)** is the value **below which 75% of the data falls**.

**Example:**
For `[10, 20, 30, 40, 50]`, Q3 is `40`.

### **Interpretation:**
- Helps understand the **upper bound** of the middle 50% of data.
- If Q3 is much higher than Q1, it means the data is **right-skewed**.

---

## 8. `max` – Maximum Value
The `max` represents the **largest value** in the column.

**Example:**
For `[10, 20, 30, 40, 50]`, the **max** is `50`.

### **Interpretation:**
- Shows the **upper bound** of the dataset.
- If the max is much higher than other values, it might be an **outlier**.

---

## **Summary Table**
| Statistic | Meaning | Formula | Interpretation |
|-----------|---------|---------|---------------|
| `count` | Non-null values | Count of non-null values | Shows the number of valid entries |
| `mean` | Average value | \( \sum X_i / N \) | Central tendency, affected by outliers |
| `std` | Standard deviation | \( \sqrt{\sum (X_i - \mu)^2 / (N-1)} \) | Measures spread of data |
| `min` | Minimum value | Smallest value in dataset | Helps find the lower bound |
| `25%` | First quartile (Q1) | Value below which 25% of data falls | Lower bound of middle 50% |
| `50%` | Median (Q2) | Middle value of sorted data | Not affected by outliers |
| `75%` | Third quartile (Q3) | Value below which 75% of data falls | Upper bound of middle 50% |
| `max` | Maximum value | Largest value in dataset | Helps find the upper bound |

---

## **Final Insights**
- **Use `mean` for normally distributed data** but use **`median` for skewed data**.
- **`std` shows variation**—a higher value means the data is more spread out.
- **Quartiles (`25%`, `50%`, `75%`) help understand the distribution** of data.
- **If `max` is much higher than `75%`, or `min` is much lower than `25%`, there may be outliers.**

