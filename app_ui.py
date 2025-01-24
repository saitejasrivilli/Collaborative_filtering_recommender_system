import streamlit as st
import pickle
import random
import pandas as pd

# Load models
import gzip
import pickle

# Function to load compressed .gz files
def load_compressed_pickle(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

# Load models from compressed .gz files
train_matrix = load_compressed_pickle("train_matrix.pkl.gz")
user_similarity_df = load_compressed_pickle("user_similarity_df.pkl.gz")
item_similarity_df = load_compressed_pickle("item_similarity_df.pkl.gz")
user_mapping = load_compressed_pickle("user_mapping.pkl.gz")
product_mapping = load_compressed_pickle("product_mapping.pkl.gz")

print("Compressed files loaded successfully!")

# Function to recommend products
def recommend_products_hybrid(user_id, train_matrix, user_similarity_df, item_similarity_df, alpha=0.5, top_n=5):
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
    hybrid_score = hybrid_score + train_matrix.mean(axis=1)[user_id]
    hybrid_score = hybrid_score.clip(lower=1, upper=5)
    hybrid_score = hybrid_score[user_ratings == 0]
    return hybrid_score.sort_values(ascending=False).head(top_n)

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state["user_id"] = ""
if "selected_option" not in st.session_state:
    st.session_state["selected_option"] = "Manual Input"

# Streamlit App
st.title("Product Recommendation System")

# Suggested users
suggested_users = random.sample(list(train_matrix.index), 5)
user_options =  [
    f"{user_mapping[user_mapping['user_id'] == user]['user_name'].values[0]} (ID: {user})" for user in suggested_users
]

# Sidebar for user selection
st.sidebar.header("Suggested User IDs")
selected_option = st.sidebar.radio("Select a Suggested User or Manual Input:", 
                                   user_options,
                                   #index= 0, 
                                   key="selected_option")

# Display the currently selected option
st.write("Currently Selected Option:")
st.text(selected_option)

if 'user_id' not in st.session_state:
    st.session_state['user_id'] = ''
if 'selected_user_id' not in st.session_state:
    st.session_state['selected_user_id'] = None


# Update session state based on the selected option
if selected_option != "Manual Input":
    try:
        selected_user_id = selected_option.split("(ID: ")[-1][:-1]  # Extract the user ID
        if st.session_state["selected_user_id"] != selected_user_id:
            # Update session state only if the selected user ID is different
            st.session_state["user_id"] = selected_user_id
            st.session_state["selected_user_id"] = selected_user_id
    except IndexError:
        st.error("Invalid user selection format")
else:
    st.session_state["selected_user_id"] = None # Clear the user ID for manual input

# Input field for User ID
user_id = st.text_input(
    "Enter User ID:",
    value=st.session_state.get("user_id",''),  # Dynamically update the field based on selection
    key="user_id_input",
)

# Update session state from manual input
if user_id and user_id != st.session_state.get("user_id"):
    st.session_state["user_id"] = user_id
    st.session_state['selected_user_id'] = None


# Recommendation settings
alpha = st.slider("Set Hybrid Alpha (0.0 = Item-Based, 1.0 = User-Based):", 0.0, 1.0, 0.5)
top_n = st.number_input("Number of Recommendations:", min_value=1, max_value=20, value=5)

# Get recommendations on button click
if st.button("Get Recommendations"):
    if st.session_state["user_id"]:
        recommendations = recommend_products_hybrid(
            st.session_state["user_id"], train_matrix, user_similarity_df, item_similarity_df, alpha, top_n
        )
        if isinstance(recommendations, str):
            st.error("User not found!")
        else:
            st.write("Recommended Products:")
            for asin, score in recommendations.items():
                product_name = product_mapping[product_mapping["asin"] == asin]["product_name"].values[0]
                st.write(f"{product_name} (ASIN: {asin}) - Predicted Rating: {score}")
    else:
        st.error("Please enter a valid User ID.")
