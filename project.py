import tkinter as tk
from tkinter import filedialog
import pandas as pd
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Create the main window
window = tk.Tk()

# Function to perform sentiment analysis and generate the final report
def generate_report():
    # Load the data from the selected Excel file
    file_path = filedialog.askopenfilename(filetypes=[('Excel Files', '*.xlsx')])
    data = pd.read_excel(file_path)

    # Clean the text data
    data['clean_text'] = data['reviews'].apply(lambda x: str(x).lower())

    # Perform sentiment analysis
    sia = SentimentIntensityAnalyzer()
    data['sentiment_scores'] = data['clean_text'].apply(lambda x: sia.polarity_scores(x))
    data['compound_score'] = data['sentiment_scores'].apply(lambda x: x['compound'])
    data['sentiment'] = data['compound_score'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['clean_text'])

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X, data['sentiment'])

    # Assume you have a new dataset for future business sentiment prediction
    future_file_path = filedialog.askopenfilename(filetypes=[('Excel Files', '*.xlsx')])
    future_data = pd.read_excel(future_file_path)

    # Clean the text data in the future dataset
    future_data['clean_text'] = future_data['reviews'].apply(lambda x: str(x).lower())

    # Vectorize the future text data using the same TF-IDF vectorizer
    X_future = vectorizer.transform(future_data['clean_text'])

    # Make predictions for future sentiment
    future_data['predicted_sentiment'] = model.predict(X_future)

    # Update the final report based on the specified conditions
    def update_turnout(sentiment):
        if sentiment == 'Positive':
            return 'high'
        elif sentiment == 'Neutral':
            return 'acceptable'
        elif sentiment == 'Negative':
            return 'weak'
        else:
            return 'unknown'

    final_report = pd.DataFrame({
        'hotel_name': data['hotel_name'],
        'reviews': data['reviews'],
        'classify_sentiment': data['sentiment'],
        'predicted_future_business': future_data['predicted_sentiment'].apply(update_turnout)
    })

    # Save the final report to an Excel sheet
    final_report.to_excel('final_report.xlsx', index=False)

    # Show a message box when the report generation is complete
    tk.messagebox.showinfo('Report Generated', 'باي من غير سلام')

# Create a button to trigger the report generation
generate_button = tk.Button(window, text='يلا بينا', command=generate_report)
generate_button.pack()

window.mainloop()