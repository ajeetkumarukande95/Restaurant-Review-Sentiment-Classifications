from flask import Flask, render_template, request, jsonify
from predict import predict_sentiment

app = Flask(__name__)

# Serve the index.html file
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint for predicting sentiment
@app.route('/predict_sentiment', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review_text = data['review']
        sentiment = predict_sentiment(review_text)
        return jsonify({'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
