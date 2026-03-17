-- Display first 5 rows of the customers table
SELECT * FROM customers
LIMIT 5;

-- Churn rate for each contract type
SELECT Contract,
       ROUND(AVG(Churn='Yes')*100,2) AS Churn_Rate_Percent,
       COUNT(*) AS Number_of_Customers
FROM customers
GROUP BY Contract;

-- Average Tenure and MonthlyCharges by churn status
SELECT Churn,
       ROUND(AVG(Tenure),1) AS Average_Tenure,
       ROUND(AVG(MonthlyCharges),2) AS Average_Monthly_Charges
FROM customers
GROUP BY Churn;

-- Show top 10 customers with high churn probability (>0.7)
SELECT CustomerID, Contract, Tenure, MonthlyCharges, ChurnProbability
FROM customers
WHERE ChurnProbability > 0.7
ORDER BY ChurnProbability DESC
LIMIT 10;


-- Count number of customers in each risk level
SELECT RiskLevel, COUNT(*) AS NumCustomers
FROM CustomerRisk
GROUP BY RiskLevel;

-- Cohort analysis: group customers by tenure ranges and calculate churn rate
SELECT 
    CASE 
        WHEN Tenure <= 12 THEN '0-12 months'
        WHEN Tenure <= 24 THEN '13-24 months'
        ELSE '25+ months'
    END AS TenureGroup,
    ROUND(AVG(Churn='Yes')*100,2) AS ChurnRatePercent,
    COUNT(*) AS NumCustomers
FROM customers
GROUP BY TenureGroup;
