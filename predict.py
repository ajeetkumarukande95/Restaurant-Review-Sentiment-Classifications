import re
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')

# Load trained model and CountVectorizer
model_path = 'sentiment_model'
cv_path = 'Countvectorizer.pkl'
classifier = joblib.load(model_path)
cv = joblib.load(cv_path)

# Initialize WordNetLemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

def predict_sentiment(new_review):
    # Preprocess the new review text
    def preprocess_text(text):
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text if word not in set(all_stopwords)]
        text = ' '.join(text)
        return text

    processed_review = preprocess_text(new_review)

    # Transform the processed review using the CountVectorizer
    transformed_review = cv.transform([processed_review]).toarray()

    # Predict sentiment
    predicted_sentiment = classifier.predict(transformed_review)

    # Interpret the predicted sentiment
    if predicted_sentiment[0] == 1:
        return "Positive"
    else:
        return "Negative"
