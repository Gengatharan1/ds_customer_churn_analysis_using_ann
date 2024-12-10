# Churn Analysis Report

## 1. **Dataset Overview**
The dataset contains information about 10,000 customers and their banking details, used to predict whether a customer has churned. Key attributes include customer demographics, account information, and churn status.

### Features Summary
- **RowNumber**: Customer index.
- **CustomerId**: Unique identifier for each customer.
- **Surname**: Customer's surname.
- **CreditScore**: Credit score of the customer.
- **Geography**: Country of residence.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Tenure**: Number of years the customer has been with the bank.
- **Balance**: Account balance.
- **NumOfProducts**: Number of products the customer has with the bank.
- **HasCrCard**: Whether the customer has a credit card (1 = Yes, 0 = No).
- **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No).
- **EstimatedSalary**: Estimated salary of the customer.
- **Exited**: Target variable indicating whether the customer has churned (1 = Yes, 0 = No).

## 2. **Data Exploration**

### Missing Values
- **No missing values** were identified in the dataset.

### Basic Statistics
- **Credit Score**: Mean = 650, Min = 350, Max = 850.
- **Age**: Mean = 39, Min = 18, Max = 92.
- **Balance**: Mean = 76,486, Min = 0, Max = 250,898.
- **Churn Rate**: 20.37% of customers have churned.

## 3. **Data Cleaning and Encoding**

### Encoding
- The categorical features `Geography` and `Gender` were encoded:
  - Created dummy variables: `Geography_France`, `Geography_Germany`, `Geography_Spain`, `Gender_Male`, and `Gender_Female`.

## 4. **Visualization and Insights**

### Churn Distribution
- Approximately **20.37%** of customers have churned.

![Churn Rate Visualization](path-to-visualization.img)

### Correlation Analysis
- **Age**: Positively correlated with churn, indicating older customers are more likely to churn.
- **IsActiveMember**: Negatively correlated with churn; active members are less likely to churn.
- **Balance**: Slightly positively correlated with churn.

![Correlation Matrix](path-to-correlation-matrix.img)

## 5. **Insights and Recommendations**

### Key Insights
1. **Older Customers at Higher Risk**:
   - The analysis indicates that churn rates increase with age. Older customers may have unique needs that are not being met, leading to dissatisfaction and eventual churn.

2. **Inactive Members Are Vulnerable**:
   - Customers who are not actively engaging with the bank are significantly more likely to leave. This suggests a need to enhance member engagement.

3. **Regional Trends**:
   - Geography plays a role, with German customers showing higher churn rates compared to customers from Spain and France.

### Actionable Recommendations
1. **Personalized Retention Strategies for Older Customers**:
   - Introduce age-specific loyalty programs such as higher interest savings accounts or exclusive financial planning services.

2. **Enhance Member Engagement**:
   - Develop campaigns to promote active usage of products and services. For example, reward programs for frequent transactions or exclusive benefits for active members.

3. **Regional Customization**:
   - Conduct deeper analysis to identify reasons behind the high churn in Germany. Tailor marketing campaigns, improve regional service offerings, or offer exclusive deals to address local customer needs.

4. **Predictive Monitoring**:
   - Implement churn prediction models to identify at-risk customers and deploy proactive retention efforts.