from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv('books.csv')

# Convert the 'publication_date' column to datetime format and drop rows with NaT
df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
df.dropna(subset=['publication_date'], inplace=True)
df['publication_date'] = df['publication_date'].apply(lambda x: x.timestamp())

# List of categorical columns to encode
categorical_columns = ['title', 'authors', 'language_code', 'publisher']

# Initialize the LabelEncoder
label_encoders = {col: LabelEncoder() for col in categorical_columns}

# Apply label encoding to the categorical columns
for column in categorical_columns:
    df[column] = label_encoders[column].fit_transform(df[column])

# Columns to convert to numeric
columns = ['bookID', 'average_rating', 'isbn', 'isbn13', 'num_pages', 'ratings_count', 'text_reviews_count']

# Convert 'isbn' and 'isbn13' to numeric if they are not already (here assuming they are strings)
df['isbn'] = pd.to_numeric(df['isbn'], errors='coerce')
df['isbn13'] = pd.to_numeric(df['isbn13'], errors='coerce')

# Drop rows with NaN values in the specified columns
df.dropna(subset=columns, inplace=True)

# Standardize numerical features
scaler = StandardScaler()
df[['average_rating', 'num_pages', 'ratings_count', 'text_reviews_count']] = scaler.fit_transform(df[['average_rating', 'num_pages', 'ratings_count', 'text_reviews_count']])

# Create a mapping between bookID and title
id_to_title = dict(zip(df['bookID'], df['title']))

# Calculate cosine similarity matrix
features = ['authors', 'average_rating', 'language_code', 'num_pages', 'ratings_count', 'text_reviews_count', 'publisher']
cosine_sim = cosine_similarity(df[features])

# Function to recommend books based on content features
def recommend_books(book_id, cosine_sim, df, id_to_title, label_encoders, top_n=5):
    # Get the index of the book that matches the book_id
    idx = df.index[df['bookID'] == book_id][0]

    # Get the pairwise similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top_n most similar books
    sim_scores = sim_scores[1:top_n+1]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top_n most similar books
    recommended_books = [label_encoders['title'].inverse_transform([id_to_title[df.iloc[i]['bookID']]])[0] for i in book_indices]
    return recommended_books

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    book_id = int(request.form['book_id'])
    recommendations = recommend_books(book_id, cosine_sim, df, id_to_title, label_encoders)
    return render_template('result.html', recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
