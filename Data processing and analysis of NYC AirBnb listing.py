import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset into a pandas DataFrame
data: DataFrame = pd.read_csv(""D:\AB_NYC_2019.csv"")

# Handling missing values
# Let's fill missing values in 'reviews_per_month' with the mean value
impute = SimpleImputer(strategy='mean')
data['reviews_per_month'] = impute.fit_transform(data[['reviews_per_month']])

# Data preprocessing
# Define numerical and categorical features
numeric_features = ['latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                    'calculated_host_listings_count', 'availability_365']
categorical_features = ['neighbourhood_group', 'room_type']

# Define preprocessing steps for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler())  # Standardize numerical features
])

categorical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing steps to the entire dataset
transformed_data = preprocessor.fit_transform(data)

# Feature engineering
# Let's create a new feature 'price_per_review' as the ratio of 'price' to 'number_of_reviews'
data['price_per_review'] = data['price'] / data['number_of_reviews']

# Explore the dataset
print(data.head())  # Display the first few rows of the dataset
print(data.info())  # Display information about the dataset, including data types and missing values
print(data.describe())  # Display basic statistics of numerical columns

# Analyze different aspects of the dataset

# What can we learn about different hosts and areas?

# Number of unique hosts
num_hosts = data['host_id'].nunique()
print("Number of unique hosts:", num_hosts)

# Number of unique neighbourhoods
num_neighbourhoods = data['neighbourhood'].nunique()
print("Number of unique neighbourhoods:", num_neighbourhoods)

# What can we learn from predictions?

# Analyze prices
avg_price = data['price'].mean()
max_price = data['price'].max()
min_price = data['price'].min()
print("Average price:", avg_price)
print("Maximum price:", max_price)
print("Minimum price:", min_price)

# Analyze reviews
avg_reviews_per_month = data['reviews_per_month'].mean()
max_reviews_per_month = data['reviews_per_month'].max()
min_reviews_per_month = data['reviews_per_month'].min()
print("Average reviews per month:", avg_reviews_per_month)
print("Maximum reviews per month:", max_reviews_per_month)
print("Minimum reviews per month:", min_reviews_per_month)

# Which hosts are the busiest and why?

# Top 10 hosts with the most listings
top_hosts = data['host_id'].value_counts().head(10)
print("Top 10 busiest hosts:\n", top_hosts)

# Is there any noticeable difference of traffic among different areas and what could be the reason for it?

# Analyze traffic among different neighbourhoods
neighbourhood_traffic = data['neighbourhood'].value_counts()
print("Traffic among different neighbourhoods:\n", neighbourhood_traffic)

# Visualize the data and draw conclusions

# What can we learn about different hosts and areas?

# Distribution of hosts across different neighbourhood groups
plt.figure(figsize=(10, 6))
sns.countplot(x='neighbourhood_group', data=data)
plt.title('Distribution of Hosts Across Different Neighbourhood Groups')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Number of Hosts')
plt.show()

# What can we learn from predictions?

# Analyze prices
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], bins=50, kde=True)
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Analyze reviews per month
plt.figure(figsize=(10, 6))
sns.histplot(data['reviews_per_month'].dropna(), bins=50, kde=True)
plt.title('Distribution of Reviews per Month')
plt.xlabel('Reviews per Month')
plt.ylabel('Frequency')
plt.show()

# Which hosts are the busiest and why?

# Top 10 hosts with the most listings
top_hosts = data['host_id'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_hosts.index, y=top_hosts.values)
plt.title('Top 10 Busiest Hosts')
plt.xlabel('Host ID')
plt.ylabel('Number of Listings')
plt.xticks(rotation=45)
plt.show()

# Is there any noticeable difference of traffic among different areas and what could be the reason for it?

# Analyze traffic among different neighbourhoods
neighbourhood_traffic = data['neighbourhood'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=neighbourhood_traffic.index, y=neighbourhood_traffic.values)
plt.title('Top 10 Neighbourhoods with Highest Traffic')
plt.xlabel('Neighbourhood')
plt.ylabel('Number of Listings')
plt.xticks(rotation=45)
plt.show()
