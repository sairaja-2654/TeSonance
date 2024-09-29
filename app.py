import pandas as pd
import streamlit as st
import pickle

# Load the dataset
df = pd.read_csv("binary_with_artist.csv", low_memory=False)

# Remove unnecessary spaces in column names
df.columns = df.columns.str.strip()

# Extract necessary data
songs = df['name'].tolist()  # List of available songs
song_info = df[['name', 'mood', 'artist', 'year']]

# Load the pre-trained Logistic Regression model from the .pkl file (Linux path)
# Update this to the correct path in your Linux environment
with open("logistic_model.pkl", 'rb') as file:
    logistic_model = pickle.load(file)

# Streamlit App
st.title("TeSonance: Mood Analysis of Telugu Songs")

# Song selection dropdown
selected_song = st.selectbox("Select a song:", songs)

# Button to find mood
if st.button("Find Mood"):
    # Get the corresponding row from the dataset for the selected song
    song_details = song_info[song_info['name'] == selected_song]

    if not song_details.empty:
        # Get the artist and year
        artist = song_details['artist'].values[0]
        year = song_details['year'].values[0]

        # Extract features for prediction (make sure to match the training features)
        try:
            # Drop only the columns that exist in the dataset
            song_features = df[df['name'] == selected_song].drop(['name', 'mood', 'year', 'artist', 'lyrics'], axis=1, errors='ignore')

            # Predict the mood using the logistic regression model
            predicted_mood = logistic_model.predict(song_features)[0]

            # Display the song details
            st.subheader(f"Mood: {predicted_mood}")
            st.write(f"Artist: {artist}")
            st.write(f"Year: {year}")
        except Exception as e:
            st.error(f"Error processing song features: {e}")
    else:
        st.error("Song not found")
