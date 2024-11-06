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

% Build a linear regression model
X_train_linear = dataTrain{:, selected_features};
y_train_linear = dataTrain{:, 'price'};
mdl_linear = fitlm(X_train_linear, y_train_linear);

% Model Evaluation
X_test = dataTest{:, selected_features};
y_test = dataTest{:, 'price'};

% Make predictions and measure prediction time
tic;
y_pred_linear = predict(mdl_linear, X_test);
predict_time = toc;

% Calculate RMSE for test set
rmse_linear = sqrt(mean((y_test - y_pred_linear).^2));

% Display results
disp('Linear Regression Metrics:');
disp(['Test RMSE: ', num2str(rmse_linear)]);

% Scatter plot for actual vs. predicted prices
figure;
scatter(y_test, y_pred_linear);
title('Linear Regression: Actual vs. Predicted Prices');
xlabel('Actual Prices');
ylabel('Predicted Prices');
axis equal;
grid on;
xtickformat('%,.0f');

% Residuals plot
figure;
plotResiduals(mdl_linear, 'probability');
title('Linear Regression: Residuals Plot');

% Additional performance metrics
train_error_linear = sqrt(mean((y_train_linear - mdl_linear.predict(X_train_linear)).^2));
test_error_linear = rmse_linear;
train_time_linear = 0; % Negligible for linear regression
disp('Additional Metrics:');
disp(['Avg Train Error: ', num2str(train_error_linear)]);
disp(['Test Error: ', num2str(test_error_linear)]);
disp(['Train Time: ', num2str(train_time_linear)]);
disp(['Predict Time: ', num2str(predict_time)]);

% Check and display distribution of 'price'
price_distribution = tabulate(data.price);
figure;
bar(price_distribution(:, 1), price_distribution(:, 2));
xlabel('Price');
ylabel('Number of Instances');
title('Distribution of Price Classes');

% Scatter plot with regression line
figure;
scatter(y_test, y_pred_linear);
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k'); % Diagonal line for perfect predictions
title('Linear Regression: Actual vs. Predicted Prices');
xlabel('Actual Prices');
ylabel('Predicted Prices');
axis equal;
grid on;
xtickformat('%,.0f');
legend('Predictions', 'Perfect Predictions', 'Location', 'northwest');

% Residuals vs. Fitted values plot
figure;
plotResiduals(mdl_linear, 'fitted');
title('Linear Regression: Residuals vs. Fitted Values Plot');
xlabel('Fitted Values');
ylabel('Residuals');

% Histogram of residuals
figure;
histogram(mdl_linear.Residuals.Raw, 'BinWidth', 20);
title('Histogram of Residuals');
xlabel('Residuals');
ylabel('Frequency');

% Display Additional Metrics and Class Distribution
disp('Additional Metrics:');
disp(['Avg Train Error: ', num2str(train_error_linear)]);
disp(['Test Error: ', num2str(test_error_linear)]);
disp(['Train Time: ', num2str(train_time_linear)]);
disp(['Predict Time: ', num2str(predict_time)]);
disp('Class Distribution:');
figure;
bar(price_distribution(:, 1), price_distribution(:, 2));
xlabel('Price');
ylabel('Number of Instances');
title('Distribution of Price Classes');

% Display feature importance coefficients
disp('Feature Importance (Coefficients):');
disp(table(mdl_linear.Coefficients.Estimate(2:end), 'RowNames', ...
    selected_features', 'VariableNames', {'Coefficient'}));

% Display goodness-of-fit metrics
disp('Goodness of Fit Metrics:');
disp(['R squared: ', num2str(mdl_linear.Rsquared.Ordinary)]);
disp(['Modified R squared: ', num2str(mdl_linear.Rsquared.Adjusted)]);
