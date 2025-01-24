from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import pickle

# Load the dataset
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
df = pd.DataFrame(dataset["full"])
df = df[['user_id', 'asin', 'rating']]  # Filter relevant columns
df.dropna(subset=['user_id', 'asin', 'rating'], inplace=True)

# Convert data types
df['user_id'] = df['user_id'].astype(str)
df['asin'] = df['asin'].astype(str)
df['rating'] = df['rating'].astype(float)

# Generate user-friendly names for users
unique_user_ids = df['user_id'].unique()

def generate_user_names(count):
    first_names = ["John", "Jane", "Alice", "Bob", "Charlie", "Emily", "Michael", "Sarah"]
    last_names = ["Doe", "Smith", "Johnson", "Brown", "Davis", "Wilson", "Taylor", "Anderson"]
    return [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(count)]

user_mapping = pd.DataFrame({
    "user_id": unique_user_ids,
    "user_name": generate_user_names(len(unique_user_ids))
})

# Generate random product names for ASINs
unique_asins = df['asin'].unique()

def generate_product_names(count):
    categories = ["Beauty", "Skincare", "Haircare", "Fragrance", "Makeup"]
    adjectives = ["Amazing", "Durable", "Stylish", "Affordable", "Luxurious"]
    return [f"{random.choice(adjectives)} {random.choice(categories)} Product" for _ in range(count)]

product_mapping = pd.DataFrame({
    "asin": unique_asins,
    "product_name": generate_product_names(len(unique_asins))
})

# Merge user and product names into the dataset
df = pd.merge(df, user_mapping, on="user_id", how="left")
df = pd.merge(df, product_mapping, on="asin", how="left")

# Filter users and items with minimum interactions
min_reviews = 5
filtered_users = df['user_id'].value_counts()[df['user_id'].value_counts() >= min_reviews].index
filtered_items = df['asin'].value_counts()[df['asin'].value_counts() >= min_reviews].index
df = df[df['user_id'].isin(filtered_users) & df['asin'].isin(filtered_items)]

# Split into training and testing data
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
train_data = train_data.groupby(['user_id', 'asin'], as_index=False)['rating'].mean()  # Aggregate ratings

# Create the User-Item Matrix
train_matrix = train_data.pivot(index='user_id', columns='asin', values='rating').fillna(0)
print(f"Train Matrix Shape: {train_matrix.shape}")

# Compute similarities
user_similarity = cosine_similarity(train_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=train_matrix.index, columns=train_matrix.index)

item_similarity = cosine_similarity(train_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=train_matrix.columns, columns=train_matrix.columns)

# Save matrices for later use
with open("train_matrix.pkl", "wb") as f:
    pickle.dump(train_matrix, f)
with open("user_similarity_df.pkl", "wb") as f:
    pickle.dump(user_similarity_df, f)
with open("item_similarity_df.pkl", "wb") as f:
    pickle.dump(item_similarity_df, f)
with open("user_mapping.pkl", "wb") as f:
    pickle.dump(user_mapping, f)
with open("product_mapping.pkl", "wb") as f:
    pickle.dump(product_mapping, f)

print("Model components and mappings saved!")

# Define recommendation functions
def recommend_products_hybrid(user_id, train_matrix, user_similarity_df, item_similarity_df, alpha=0.5, top_n=5, scale_factor=10):
    if user_id not in train_matrix.index:
        return "User not found!"
    user_similarity_df = user_similarity_df.reindex(index=train_matrix.index, columns=train_matrix.index)
    item_similarity_df = item_similarity_df.reindex(index=train_matrix.columns, columns=train_matrix.columns)
    user_ratings = train_matrix.loc[user_id].reindex(train_matrix.columns, fill_value=0)
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    user_based_score = pd.Series(0.0, index=train_matrix.columns)
    if similar_users.sum() > 0:
        aligned_train_matrix = train_matrix.reindex(index=similar_users.index, columns=train_matrix.columns)
        user_based_score = similar_users.dot(aligned_train_matrix) / similar_users.sum()
    item_based_score = pd.Series(0.0, index=train_matrix.columns)
    for item in train_matrix.columns:
        if item_similarity_df[item].sum() > 0:
            item_based_score[item] = item_similarity_df[item].dot(user_ratings) / item_similarity_df[item].sum()
    hybrid_score = alpha * user_based_score + (1 - alpha) * item_based_score
    hybrid_score += train_matrix.mean(axis=1)[user_id]
    hybrid_score *= scale_factor
    hybrid_score = hybrid_score.clip(lower=1, upper=5)
    hybrid_score = hybrid_score[user_ratings == 0]
    return hybrid_score.sort_values(ascending=False).head(top_n)

# Evaluate RMSE
predictions = []
true_ratings = []
for index, row in test_data.iterrows():
    user_id = row['user_id']
    product_id = row['asin']
    true_ratings.append(row['rating'])
    if user_id in train_matrix.index and product_id in train_matrix.columns:
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
        user_based_score = 0
        if similar_users.sum() > 0:
            aligned_ratings = train_matrix[product_id].reindex(similar_users.index, fill_value=0)
            user_based_score = similar_users.dot(aligned_ratings) / similar_users.sum()
        item_based_score = 0.0
        if item_similarity_df[product_id].sum() > 0:
            user_ratings = train_matrix.loc[user_id].reindex(train_matrix.columns, fill_value=0)
            item_based_score = item_similarity_df[product_id].dot(user_ratings) / item_similarity_df[product_id].sum()
        predicted_rating = 0.5 * user_based_score + 0.5 * item_based_score
        predicted_rating += train_matrix.mean(axis=1)[user_id]
        predicted_rating = max(1, min(5, predicted_rating))
        predictions.append(predicted_rating)
    else:
        predictions.append(np.nan)
true_ratings = np.array(true_ratings)[~np.isnan(predictions)]
predictions = np.array(predictions)[~np.isnan(predictions)]
rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
print(f"RMSE: {rmse}")

