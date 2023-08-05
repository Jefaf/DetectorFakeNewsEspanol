from flask import Flask, render_template, request, jsonify
import nltk
import pickle
import re
import unidecode

from sklearn.ensemble import RandomForestClassifier
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model2.pkl', 'rb'))
tfidfvect = pickle.load(open('tfidfvect2.pkl', 'rb'))

# Build functionalities
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

def predict(content):

    preprocessed_content = re.sub('[^a-zA-ZñÑáéíóú]',' ',content)
    preprocessed_content = unidecode.unidecode(preprocessed_content)
    preprocessed_content = preprocessed_content.lower()
    preprocessed_content = preprocessed_content.split()
    preprocessed_content = [word for word in preprocessed_content if not word in stopwords.words('spanish')]
    preprocessed_content = ' '.join(preprocessed_content)
    nlp = spacy.load('es_core_news_md')
    document = nlp(preprocessed_content)
    preprocessed_content = ''
    for token in document:
      preprocessed_content = preprocessed_content + token.lemma_+ " "
    
    preprocessed_content = tfidfvect.transform([preprocessed_content])
    prediction = model.predict(preprocessed_content)
    prediction = 'Noticia FALSA' if model.predict(preprocessed_content) == 0 else 'Noticia REAL'
    return prediction


@app.route('/', methods=['POST'])
def webapp():
    
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)


@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)


if __name__ == "__main__":
    app.run()