import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 1. Load Dataset
# We use the classic Iris dataset: predicting flower species based on measurements
df = sns.load_dataset('iris')

print("--- Data Snapshot ---")
print(df.head())

# ==========================================
# PART A: Visualization (EDA)
# ==========================================

# 1. Pairplot (Multivariate Analysis)
# This creates a grid of Scatter Plots for every pair of features
# and Histograms on the diagonal.
# Hue='species' colors the points by class, helping us see separability.
sns.pairplot(df, hue='species', height=2.5)
plt.suptitle('Pairplot of Iris Features', y=1.02)
plt.show()

# 2. Correlation Heatmap (Bivariate Analysis)
# We compute the correlation matrix (Pearson's r)
# Note: We drop the 'species' column because correlation requires numbers.
plt.figure(figsize=(8, 6))
corr_matrix = df.drop('species', axis=1).corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Interpretation:
# If you see a square with 0.96 (like petal_length vs petal_width),
# those two features give almost the same info. You might drop one (Dimensionality Reduction!).

# 3. Box Plot (Univariate/Bivariate)
# Visualizing the distribution of 'sepal_length' across different species
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='sepal_length', data=df)
plt.title('Distribution of Sepal Length by Species')
plt.show()

# ==========================================
# PART B: Statistical Tests
# ==========================================

print("\n--- Statistical Hypothesis Testing ---\n")

# 1. Normality Test (Shapiro-Wilk)
# Assumption Check: Many tests (like t-test) assume data is Normal (Gaussian).
# H0 (Null): The data IS normally distributed.
# H1 (Alt): The data IS NOT normally distributed.

setosa_sepal = df[df['species'] == 'setosa']['sepal_width']
stat, p_value = stats.shapiro(setosa_sepal)

print(f"Shapiro-Wilk Test for Setosa Sepal Width:")
print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")

if p_value > 0.05:
    print("Result: p > 0.05. We CANNOT reject H0. Data looks Normal. (Safe to use t-test)")
else:
    print("Result: p < 0.05. Reject H0. Data is NOT Normal. (Consider Non-parametric tests like Mann-Whitney)")


# 2. T-Test (Independent Two-Sample)
# Scenario: Is there a significant difference in 'sepal_width' 
# between 'setosa' and 'versicolor' species?
# H0: Means are EQUAL.
# H1: Means are DIFFERENT.

group1 = df[df['species'] == 'setosa']['sepal_width']
group2 = df[df['species'] == 'versicolor']['sepal_width']

# We use equal_var=False (Welch's t-test) just to be safe in case variances differ
t_stat, p_val_ttest = stats.ttest_ind(group1, group2, equal_var=False)

print(f"\nT-Test (Setosa vs Versicolor Sepal Width):")
print(f"T-statistic: {t_stat:.4f}, p-value: {p_val_ttest:.4e}") # .4e is scientific notation

if p_val_ttest < 0.05:
    print("Result: p < 0.05. Reject H0. There IS a significant difference between the groups.")
else:
    print("Result: p > 0.05. Cannot reject H0. No significant difference found.")

# 3. ANOVA (Analysis of Variance)
# Scenario: Compare 'sepal_width' across ALL THREE species at once.
# H0: All group means are EQUAL.
# H1: At least one group mean is DIFFERENT.

group3 = df[df['species'] == 'virginica']['sepal_width']

f_stat, p_val_anova = stats.f_oneway(group1, group2, group3)

print(f"\nANOVA (Comparing all 3 species):")
print(f"F-statistic: {f_stat:.4f}, p-value: {p_val_anova:.4e}")

if p_val_anova < 0.05:
    print("Result: p < 0.05. Reject H0. The species are statistically different.")