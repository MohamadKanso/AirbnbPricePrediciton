%*NOTE*
% This code represents the initial stages of my project. It includes data
% preprocessing, model building, and preliminary evaluation. This code
% serves as a foundational reference for the development of the project.

% Load the dataset
data = readtable('AB_US_2023.csv');

% Data Preprocessing
% Normalize numerical features for consistency
numeric_vars = ["latitude", "longitude", "minimum_nights", "number_of_reviews", ...
    "reviews_per_month", "calculated_host_listings_count", "availability_365"];
data{:, numeric_vars} = normalize(data{:, numeric_vars});

% Handle date columns
data.last_review = datetime(data.last_review, 'InputFormat', 'dd/MM/yyyy');

% Select features for the model
selected_features = ["latitude", "longitude", "minimum_nights", "availability_365"];

% Split the dataset into training and testing sets
rng(42);
cv = cvpartition(size(data, 1), 'Holdout', 0.2);
dataTrain = data(training(cv), :);
dataTest = data(test(cv), :);

% Build a linear regression model
X_train_linear = dataTrain{:, selected_features};
y_train_linear = dataTrain{:, 'price'};
mdl_linear = fitlm(X_train_linear, y_train_linear);

% Build a Random Forest Model
X_train_rf = dataTrain{:, selected_features};
y_train_rf = dataTrain{:, 'price'};
mdl_rf = TreeBagger(50, X_train_rf, y_train_rf, 'Method', 'regression', ...
    'OOBPredictorImportance', 'on');

% Model evaluation
X_test = dataTest{:, selected_features};
y_test = dataTest{:, 'price'};

% Make predictions with the linear regression model
y_pred_linear = predict(mdl_linear, X_test);

% Make predictions with the random forest model
y_pred_rf = predict(mdl_rf, X_test);

% Calculate RMSE for each model
rmse_linear = sqrt(mean((y_test - y_pred_linear).^2));
rmse_rf = sqrt(mean((y_test - y_pred_rf).^2));

% Display RMSE results
disp(['RMSE for Linear Regression: ', num2str(rmse_linear)]);
disp(['RMSE for Random Forest: ', num2str(rmse_rf)]);

% Bar plot to compare RMSE
figure;
bar([rmse_linear, rmse_rf]);
title('Comparison of RMSE between Linear Regression and Random Forest');
xticklabels({'Linear Regression', 'Random Forest'});
ylabel('Root Mean Squared Error (RMSE)');
