from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load your dataset
df = pd.read_csv('books.csv')

# Preprocessing steps
df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
df.dropna(subset=['publication_date'], inplace=True)
df['publication_date'] = df['publication_date'].apply(lambda x: x.timestamp())

label_encoder_authors = LabelEncoder()
df['authors'] = label_encoder_authors.fit_transform(df['authors'])

onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
onehot_encoded_features = onehot_encoder.fit_transform(df[['language_code', 'publisher']])
onehot_feature_names = onehot_encoder.get_feature_names_out(['language_code', 'publisher'])

numerical_features = ['average_rating', 'num_pages', 'ratings_count', 'text_reviews_count']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

X = np.hstack((df[['authors']].values, df[numerical_features].values, onehot_encoded_features))
df_encoded = pd.DataFrame(X, columns=['authors'] + numerical_features + list(onehot_feature_names))

id_to_title = dict(zip(df['bookID'], df['title']))

features = ['authors'] + numerical_features + list(onehot_feature_names)

cosine_sim = cosine_similarity(df_encoded[features])

def recommend_books(input_data, df_encoded, cosine_sim, label_encoder_authors, onehot_encoder, top_n=5):
    # Encode input data
    authors_encoded = label_encoder_authors.transform([input_data[0]]).reshape(-1, 1)
    numerical_input = np.array(input_data[1:5]).reshape(1, -1)
    numerical_input_scaled = scaler.transform(numerical_input)
    onehot_encoded_features = onehot_encoder.transform(np.array(input_data[5:7]).reshape(1, -1))

    # Combine all features into a single input array
    combined_input = np.hstack((authors_encoded, numerical_input_scaled, onehot_encoded_features))

    # Create a temporary DataFrame to compute similarity
    temp_df = df_encoded[features].copy()
    temp_df.loc[len(temp_df)] = combined_input.flatten()

    # Compute cosine similarity for the new input
    new_cosine_sim = cosine_similarity(temp_df)

    # Get the pairwise similarity scores of all books with the new input
    sim_scores = list(enumerate(new_cosine_sim[-1]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores[:-1], key=lambda x: x[1], reverse=True)

    # Get the scores of the top_n most similar books
    sim_scores = sim_scores[:top_n]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top_n most similar books
    recommended_books = [id_to_title[df.iloc[idx]['bookID']] for idx in book_indices]
    return recommended_books

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    authors = request.form['authors']
    average_rating = float(request.form['average_rating'])
    num_pages = int(request.form['num_pages'])
    ratings_count = int(request.form['ratings_count'])
    text_reviews_count = int(request.form['text_reviews_count'])
    language_code = request.form['language_code']
    publisher = request.form['publisher']
    
    input_data = [authors, average_rating, num_pages, ratings_count, text_reviews_count, language_code, publisher]
    recommendations = recommend_books(input_data, df_encoded, cosine_sim, label_encoder_authors, onehot_encoder)
    
    return render_template('result.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
