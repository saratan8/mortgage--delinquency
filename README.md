# mortgage--delinquency

# Mortgage Default Risk Modeling with Markov Chains and Multinomial Logistic Regression

This project analyzes **mortgage performance data** from Fannie Mae (2004–2022) to estimate the probability of **loan default**, **foreclosure**, **REO**, or **prepayment** using **multinomial logistic regression** and **Markov chain modeling**.

---

## Project Objective

Model transitions between loan performance states (e.g., current → delinquent → REO) as a **Markov process**, and estimate transition probabilities using **macroeconomic variables** and borrower-specific features.

---

## Data Summary

- **Source**: Fannie Mae Single-Family Loan Performance Data  
- **Period**: January 2004 – December 2022  
- **Volume**: Millions of monthly observations  

### Key Features:
- **Borrower info**: Credit score, debt-to-income ratio (DTI), original LTV  
- **Loan metrics**: Current interest rate, unpaid principal balance (UPB), loan age  
- **Performance status**: Current, 30/60/90+ days delinquent, in foreclosure, REO, prepaid  
- **Macroeconomic indicators**: CPI, GDP, HPI (merged monthly)

---

## Modeling Approach

### 1. **Multinomial Logistic Regression**

We define 6 mutually exclusive mortgage states:

1. Current  
2. 30-days delinquent  
3. 60-days delinquent  
4. In foreclosure  
5. REO (Real Estate Owned)  
6. Prepaid  

Choose `class 0 (Current)` as the reference.  
The model estimates:

\[
\log \left( \frac{P(y = k \mid x)}{P(y = 0 \mid x)} \right) = \beta_k^\top x
\]

Use **softmax** to compute actual probabilities:

\[
P(y = k \mid x) = \frac{e^{\beta_k^\top x}}{1 + \sum_{j=1}^{K-1} e^{\beta_j^\top x}}, \quad P(y = 0 \mid x) = \frac{1}{1 + \sum_{j=1}^{K-1} e^{\beta_j^\top x}}
\]

---

### 2. **Markov Chain Transition Modeling**

- Monthly transitions are treated as a first-order Markov process  
- Construct empirical **transition matrices** across loan states  
- Use **macro variables** and borrower characteristics as covariates for transition modeling

---

## Evaluation Metrics

- Confusion matrix of predicted vs actual states  
- Out-of-sample log-loss  
- Cumulative transition matrix visualization  
- Default risk heatmaps by credit score, LTV, macro scenario

---

## Insights

- Higher LTV and lower credit score increase transition probability to delinquency and foreclosure  
- Macroeconomic indicators such as GDP growth and HPI impact prepayment and default likelihoods  
- Transition probabilities evolve over time and vary by loan age and cohort

---

## Tools

- Python (Pandas, Scikit-learn, Statsmodels, Matplotlib)  
- Jupyter Notebook  
- Data ~8GB+, stored externally (available on request)

---

## Notes

Due to large file sizes, raw data is not hosted on GitHub. 
