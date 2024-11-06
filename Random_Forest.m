% Load the dataset
data = readtable('AB_US_2023.csv');

% Remove unnecessary column
data.neighbourhood_group = [];

% Identify and remove outliers in 'price' using the IQR method
q1 = quantile(data.price, 0.25);
q3 = quantile(data.price, 0.75);
iqr_price = iqr(data.price);
lower_bound_price = q1 - 1.5 * iqr_price;
upper_bound_price = q3 + 1.5 * iqr_price;
data = data(data.price >= lower_bound_price & data.price <= upper_bound_price, :);

% Additional filters for specific columns
data = data(data.number_of_reviews > 0, :);
data = data(data.calculated_host_listings_count < 10, :);
data = data(data.number_of_reviews < 400, :);
data = data(data.minimum_nights < 10, :);
data = data(data.reviews_per_month < 5, :);

% Handle missing values
data = rmmissing(data);

% Normalize numerical features
numeric_vars = ["latitude", "longitude", "minimum_nights", "number_of_reviews", ...
    "reviews_per_month", "calculated_host_listings_count", "availability_365"];
data{:, numeric_vars} = normalize(data{:, numeric_vars});

% Convert date column to datetime format
data.last_review = datetime(data.last_review, 'InputFormat', 'dd/MM/yyyy');

% Feature engineering
data.reviews_per_month_squared = data.reviews_per_month.^2;
selected_features = ["latitude", "longitude", "minimum_nights", ...
    "availability_365", "reviews_per_month_squared"];

% Split the dataset into training and testing sets
rng(42);
cv = cvpartition(size(data, 1), 'Holdout', 0.2);
dataTrain = data(training(cv), :);
dataTest = data(test(cv), :);

% Build a Random Forest model
X_train_rf = dataTrain{:, selected_features};
y_train_rf = dataTrain{:, 'price'};
numTrees = 100;
mdl_rf = fitensemble(X_train_rf, y_train_rf, 'Bag', numTrees, 'Tree', 'Type', 'regression');

% Model Evaluation
X_test = dataTest{:, selected_features};
y_test = dataTest{:, 'price'};

% Random Forest Prediction and timing
tic;
y_pred_rf = predict(mdl_rf, X_test);
predict_time_rf = toc;

% Calculate RMSE for the test set
rmse_rf = sqrt(mean((y_test - y_pred_rf).^2));

% Display results
disp('Random Forest Metrics:');
disp(['Test RMSE: ', num2str(rmse_rf)]);

% Scatter plot for actual vs. predicted prices
figure;
scatter(y_test, y_pred_rf);
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k');
title('Random Forest: Actual vs. Predicted Prices');
xlabel('Actual Prices');
ylabel('Predicted Prices');
axis equal;
grid on;
xtickformat('%,.0f');
legend('Predictions', 'Perfect Predictions', 'Location', 'northwest');

% Residuals plot
figure;
residuals_rf = y_test - y_pred_rf;
plot(y_pred_rf, residuals_rf, 'o');
title('Random Forest: Residuals Plot');
xlabel('Predicted Prices');
ylabel('Residuals');
grid on;

% Histogram of residuals
figure;
histogram(residuals_rf, 'BinWidth', 20);
title('Random Forest: Histogram of Residuals');
xlabel('Residuals');
ylabel('Frequency');

% Display additional metrics
disp('Additional Metrics for Random Forest:');
disp(['Test Error: ', num2str(rmse_rf)]);
disp(['Predict Time: ', num2str(predict_time_rf)]);

% Display the distribution of the 'price' variable
price_distribution = tabulate(data.price);
disp('Class Distribution:');
figure;
bar(price_distribution(:, 1), price_distribution(:, 2));
xlabel('Price');
ylabel('Number of Instances');
title('Distribution of Price Classes');

% Feature importance for Random Forest
disp('Feature Importance (Variable Importance) for Random Forest:');
importance_rf = predictorImportance(mdl_rf);
disp(table(importance_rf', 'RowNames', selected_features', 'VariableNames', {'Importance'}));

% Goodness-of-fit metrics for Random Forest
oob_pred_rf = oobPredict(mdl_rf);
ssRes = sum((y_train_rf - oob_pred_rf).^2);
ssTot = sum((y_train_rf - mean(y_train_rf)).^2);
oob_r_squared_rf_fixed = 1 - ssRes/ssTot;
disp(['Fixed OOB R squared: ', num2str(oob_r_squared_rf_fixed)]);

% Comparison section (Compare.m)
models = {'Linear Regression', 'Random Forest'};
rmse_values = [98.221, 87.9604];
prediction_times = [0.0069715, 1.7688];
r_squared = [0.045813, oob_r_squared_rf_fixed];

% Create a figure with subplots
figure('Position', [100, 100, 1000, 400]);

% RMSE comparison
subplot(1, 3, 1);
bar(rmse_values, 'b');
title('RMSE Comparison');
ylabel('RMSE');
xticks(1:2);
xticklabels(models);
text(1:length(models), rmse_values, num2str(rmse_values', '%.2f'), ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
grid on;

% R-squared comparison
subplot(1, 3, 2);
bar(r_squared, 'g');
title('R-squared Comparison');
ylabel('R-squared');
xticks(1:2);
xticklabels(models);
text(1:length(models), r_squared, num2str(r_squared', '%.4f'), ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
grid on;

% Prediction time comparison
subplot(1, 3, 3);
bar(prediction_times, 'r');
title('Prediction Time Comparison');
ylabel('Time (seconds)');
xticks(1:2);
xticklabels(models);
text(1:length(models), prediction_times, num2str(prediction_times', '%.4f'), ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
grid on;

sgtitle('Model Comparison');
