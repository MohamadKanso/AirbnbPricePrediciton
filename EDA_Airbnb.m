% Load the data - Importing the dataset for analysis
data = readtable('AB_US_2023.csv');

% Identify and display missing values in the dataset
missingValues = sum(ismissing(data));
missingValuesTable = table(data.Properties.VariableNames', missingValues', ...
    'VariableNames', {'VariableName', 'MissingCount'});
disp('Missing Values in the Dataset:');
disp(missingValuesTable);

% Visualize missing values using a heatmap
missingMatrix = ismissing(data);
numericMissingMatrix = double(missingMatrix);
figure;
colormap([0.8 0.8 0.8; 0.5 0.5 0.5]);
imagesc(numericMissingMatrix);
title('Proportion of Missing Values (Raw Data)');
xticks(1:size(data, 2));
xticklabels(data.Properties.VariableNames);
xtickangle(90);
yticks(1:size(data, 1));
yticklabels([]);
ylabel('Rows');

% Remove unnecessary column
data.neighbourhood_group = [];

% Display basic statistics
basicStatisticsTable = summary(data);
disp('Basic Statistics:');
disp(basicStatisticsTable);

% Define and filter data based on specific conditions
iqr_data = data(data.price >= quantile(data.price, 0.25) & ...
    data.price <= quantile(data.price, 0.75) & ...
    data.number_of_reviews > 0 & ...
    data.calculated_host_listings_count < 10 & ...
    data.number_of_reviews < 400 & ...
    data.minimum_nights < 10 & ...
    data.reviews_per_month < 5, :);

% Display basic statistics of filtered data
filteredBasicStatisticsTable = summary(iqr_data);
disp('Basic Statistics (Filtered Data):');
disp(filteredBasicStatisticsTable);

% Count missing values in the filtered dataset
filteredMissingValues = sum(ismissing(iqr_data));
filteredMissingValuesTable = table(iqr_data.Properties.VariableNames', ...
    filteredMissingValues', 'VariableNames', {'VariableName', 'MissingCount'});
disp('Missing Values in the Filtered Dataset:');
disp(filteredMissingValuesTable);

% Correlation analysis between numeric columns and price
numericColumns = {'latitude', 'longitude', 'price', 'minimum_nights', ...
    'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', ...
    'availability_365', 'number_of_reviews_ltm'};
correlations = corr(iqr_data{:, numericColumns}, 'Rows', 'complete');

% Visualize missing values in filtered data using a heatmap
figure;
imagesc(ismissing(iqr_data));
title('Missing Values Proportion (Filtered)');
colormap([1 1 1; 0.8 0.8 0.8]);
xticks(1:size(iqr_data, 2));
xticklabels(iqr_data.Properties.VariableNames);
xtickangle(90);
yticks(1:size(iqr_data, 1));
yticklabels([]);
ylabel('Rows');

% Box plot of price distribution in filtered data
figure;
boxplot(iqr_data.price, 'Colors', 'b');
title('Boxplot of Price (Filtered)');
ylabel('Price');

% KDE plot of price distribution
figure;
ksdensity(iqr_data.price);
title('KDE Plot of Price (Filtered)');
xlabel('Price');
ylabel('Density');

% Scatter plot of latitude vs. price
figure;
scatter(iqr_data.latitude, iqr_data.price, 10, 'filled', ...
    'MarkerFaceAlpha', 0.5, 'MarkerFaceColor', 'b');
title('Scatter Plot of Latitude vs. Price (Filtered)');
xlabel('Latitude');
ylabel('Price');

% Bar plot of price by room type
figure;
uniqueRoomTypes = unique(iqr_data.room_type);
means = grpstats(iqr_data.price, iqr_data.room_type, 'mean');
[sortedMeans, sortedIdx] = sort(means, 'descend');
bar(categorical(uniqueRoomTypes(sortedIdx)), sortedMeans);
title('Price by Room Type (Filtered)');
xlabel('Room Type');
ylabel('Price');
xtickangle(45);

% Correlation matrix visualization
figure;
imagesc(correlations);
title('Correlation Matrix (Filtered)');
colormap('jet');
colorbar;
[nrows, ncols] = size(correlations);
for i = 1:nrows
    for j = 1:ncols
        text(j, i, sprintf('%.2f', correlations(i, j)), 'HorizontalAlignment', ...
            'center', 'VerticalAlignment', 'middle', 'Color', 'w');
    end
end
xticks(1:length(numericColumns));
xticklabels(numericColumns);
yticks(1:length(numericColumns));
yticklabels(numericColumns);
xtickangle(60);

% Box plot of price by room type
figure;
boxplot(iqr_data.price, iqr_data.room_type);
title('Price Distribution by Room Type');
xlabel('Room Type');
ylabel('Price');

% Horizontal bar chart comparing prices by city
figure;
unique_cities = unique(iqr_data.city);
mean_prices_by_city = zeros(size(unique_cities));
for i = 1:length(unique_cities)
    city_prices = iqr_data.price(strcmp(iqr_data.city, unique_cities{i}));
    mean_prices_by_city(i) = mean(city_prices);
end
barh(unique_cities, mean_prices_by_city, 'b');
title('Price Comparison by City (Filtered)');
xlabel('Price');
ylabel('City');
