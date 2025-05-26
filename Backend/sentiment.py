from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Example input text (you can change this variable)
input_text = "Hey there! Everything looks perfect — just a tiny tweak: can you help me set the user's status to 'inactive' in the database? Thanks a ton!"

# Function to detect sentiment
def detect_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label']  # Could be 'POSITIVE', 'NEGATIVE', 'NEUTRAL', depending on the model

# Detect sentiment
sentiment = detect_sentiment(input_text)

# Output result
print(f"Input Text: {input_text}")
print(f"Detected Sentiment: {sentiment}")
