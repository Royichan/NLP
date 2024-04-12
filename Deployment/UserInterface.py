import streamlit as st
import requests

# Function to send features to an API for localization prediction
def sendReview(text):
    # Create the request body
    body = {"text":text}
    try:
        # Send a POST request to the localization prediction API
        response = requests.post(url="http://localhost:105/reviewPrediction", json=body)
    except requests.exceptions.ConnectionError as e:
         # Handle connection error to the API
        st.text("API Connection Failed")
        return []

    if response.status_code == 200:
        Score = response.json()['score']
        print(response.json())
        return Score
    else:
        # Handle API call failure
        st.text("API Call Failed")

def main():
     # Streamlit app title and user input section
    st.title("Review Score Prediction")
    text = st.text_input("Enter the review")
    
    # Button to trigger localization prediction
    if st.button("Score"):
        if len(text) != 0:
            score = sendReview(text) # Call the prediction function
            st.write(f"Possible review score : {score}")
        else:
            st.write(f"Input doesn't meet prerequisite, Input length is {len(text)}")
if __name__ == "__main__":
    main()
