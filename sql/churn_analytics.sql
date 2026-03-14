-- Churn Analytics (SQLite)
-- Table: customer_churn (loaded from data/churn.csv)
-- Target column: Exited (1 = churn, 0 = stay)

-- 1) Overall churn rate
SELECT
  COUNT(*) AS total_customers,
  SUM(Exited) AS churned_customers,
  ROUND(100.0 * SUM(Exited) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn;

-- 2) Churn rate by Geography
SELECT
  Geography,
  COUNT(*) AS total,
  SUM(Exited) AS churned,
  ROUND(100.0 * SUM(Exited) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn
GROUP BY Geography
ORDER BY churn_rate_pct DESC;

-- 3) Churn rate by Gender
SELECT
  Gender,
  COUNT(*) AS total,
  SUM(Exited) AS churned,
  ROUND(100.0 * SUM(Exited) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn
GROUP BY Gender
ORDER BY churn_rate_pct DESC;

-- 4) Churn rate by Age bands
WITH age_bands AS (
  SELECT
    CASE
      WHEN Age < 25 THEN 'Under 25'
      WHEN Age BETWEEN 25 AND 34 THEN '25-34'
      WHEN Age BETWEEN 35 AND 44 THEN '35-44'
      WHEN Age BETWEEN 45 AND 54 THEN '45-54'
      WHEN Age BETWEEN 55 AND 64 THEN '55-64'
      ELSE '65+'
    END AS age_band,
    Exited
  FROM customer_churn
)
SELECT
  age_band,
  COUNT(*) AS total,
  SUM(Exited) AS churned,
  ROUND(100.0 * SUM(Exited) / COUNT(*), 2) AS churn_rate_pct
FROM age_bands
GROUP BY age_band
ORDER BY churn_rate_pct DESC;

-- 5) Churn rate by IsActiveMember
SELECT
  IsActiveMember,
  COUNT(*) AS total,
  SUM(Exited) AS churned,
  ROUND(100.0 * SUM(Exited) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn
GROUP BY IsActiveMember
ORDER BY churn_rate_pct DESC;

-- 6) Churn rate by NumOfProducts
SELECT
  NumOfProducts,
  COUNT(*) AS total,
  SUM(Exited) AS churned,
  ROUND(100.0 * SUM(Exited) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn
GROUP BY NumOfProducts
ORDER BY NumOfProducts;

-- 7) Churn rate by Balance bands
WITH balance_bands AS (
  SELECT
    CASE
      WHEN Balance = 0 THEN 'Zero'
      WHEN Balance < 50000 THEN '<50k'
      WHEN Balance BETWEEN 50000 AND 100000 THEN '50k-100k'
      WHEN Balance BETWEEN 100000 AND 150000 THEN '100k-150k'
      ELSE '150k+'
    END AS balance_band,
    Exited
  FROM customer_churn
)
SELECT
  balance_band,
  COUNT(*) AS total,
  SUM(Exited) AS churned,
  ROUND(100.0 * SUM(Exited) / COUNT(*), 2) AS churn_rate_pct
FROM balance_bands
GROUP BY balance_band
ORDER BY churn_rate_pct DESC;

-- 8) Churn rate by CreditScore bands
WITH score_bands AS (
  SELECT
    CASE
      WHEN CreditScore < 500 THEN '<500'
      WHEN CreditScore BETWEEN 500 AND 599 THEN '500-599'
      WHEN CreditScore BETWEEN 600 AND 699 THEN '600-699'
      WHEN CreditScore BETWEEN 700 AND 799 THEN '700-799'
      ELSE '800+'
    END AS score_band,
    Exited
  FROM customer_churn
)
SELECT
  score_band,
  COUNT(*) AS total,
  SUM(Exited) AS churned,
  ROUND(100.0 * SUM(Exited) / COUNT(*), 2) AS churn_rate_pct
FROM score_bands
GROUP BY score_band
ORDER BY churn_rate_pct DESC;

-- 9) Churn rate by Tenure
SELECT
  Tenure,
  COUNT(*) AS total,
  SUM(Exited) AS churned,
  ROUND(100.0 * SUM(Exited) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn
GROUP BY Tenure
ORDER BY Tenure;

-- 10) High-risk segment example: older + inactive + high balance
SELECT
  COUNT(*) AS total,
  SUM(Exited) AS churned,
  ROUND(100.0 * SUM(Exited) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn
WHERE Age >= 45
  AND IsActiveMember = 0
  AND Balance >= 100000;
