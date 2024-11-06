# 🏠 Advanced Machine Learning in Airbnb Price Prediction
![MATLAB](https://img.shields.io/badge/MATLAB-R2023a-orange.svg)
![Data Science](https://img.shields.io/badge/Data_Science-Analysis-blue.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📊 Project Overview

An advanced machine learning analysis comparing Linear Regression and Random Forest models for Airbnb price prediction, achieving a 10% improvement in prediction accuracy over baseline models.

### 🎯 Key Achievements
- Achieved 87.96 RMSE with Random Forest implementation
- Reduced prediction error by ~10% compared to baseline
- Processed and analyzed 232,147 property listings
- Engineered novel features for enhanced prediction accuracy

## 🔬 Technical Implementation

### Data Processing
```matlab
% Data preprocessing pipeline
data = readtable('AB_US_2023.csv');
data.neighbourhood_group = [];  % Remove unnecessary column

% IQR-based outlier removal
q1 = quantile(data.price, 0.25);
q3 = quantile(data.price, 0.75);
iqr_price = iqr(data.price);
lower_bound_price = q1 - 1.5 * iqr_price;
upper_bound_price = q3 + 1.5 * iqr_price;

% Data filtering
filtered_data = data(data.price >= lower_bound_price & ...
                    data.price <= upper_bound_price & ...
                    data.number_of_reviews > 0 & ...
                    data.calculated_host_listings_count < 10 & ...
                    data.number_of_reviews < 400 & ...
                    data.minimum_nights < 10 & ...
                    data.reviews_per_month < 5, :);
```

## 📈 Model Performance

| Metric | Linear Regression | Random Forest |
|--------|------------------|---------------|
| RMSE | 98.221 | 87.960 |
| R² | 0.046 | 0.228 |
| Prediction Time (s) | 0.014 | 1.548 |

## 🛠️ Implementation Details

### Linear Regression Model
- Feature selection focused on key price indicators
- Implemented using MATLAB's fitlm function
- Fast prediction time (0.014s)
- RMSE of 98.221

### Random Forest Model
- 100 decision trees in ensemble
- Advanced feature engineering
- Robust against outliers
- RMSE of 87.960

## 📊 Key Features Analyzed
- Location (latitude, longitude)
- Property characteristics
- Review metrics
- Availability patterns
- Host performance indicators

## 🔍 Methodology

### Data Preprocessing Steps
1. Missing value treatment
2. Outlier removal using IQR
3. Feature normalization
4. Date formatting
5. Feature engineering

### Model Development
```matlab
% Feature preparation
numeric_vars = ["latitude", "longitude", "minimum_nights", ...
                "reviews_per_month", "calculated_host_listings_count", ...
                "availability_365"];
data{:, numeric_vars} = normalize(data{:, numeric_vars});

% Model training
selected_features = ["latitude", "longitude", "minimum_nights", ...
                    "availability_365", "reviews_per_month_squared"];
```

## 📈 Results Analysis

### Model Comparison
- **Linear Regression**
  - Simple implementation
  - Fast execution time
  - Limited by linear assumptions

- **Random Forest**
  - Better accuracy
  - Handles non-linear relationships
  - More computationally intensive

## 🚀 Future Enhancements

1. **Model Optimization**
   - Hyperparameter tuning
   - Cross-validation implementation
   - Advanced feature engineering

2. **Feature Engineering**
   - Geographic clustering
   - Temporal analysis
   - Review sentiment analysis

3. **Performance Improvements**
   - Parallel processing
   - Memory optimization
   - Computation efficiency

## 📚 Repository Structure

```
airbnb-price-prediction/
├── models/
│   ├── linear_regression.m
│   └── random_forest.m
├── analysis/
│   ├── EDA.m
└── README.md
```


## 📮 Contact

For inquiries or collaboration:
- 📧 Email: mohamadghorikanso@gmail.com
- 💼 LinkedIn: [Mohamad Kanso](https://www.linkedin.com/in/mohamad-kanso/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Note: This project was completed as part of my MSc Data Science program at City, University of London.*
