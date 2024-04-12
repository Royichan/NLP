from flask import Flask, make_response, request, jsonify
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


app = Flask(__name__)

#load the model
file = os.getcwd()+"\\"
tfidfVectorizer = pickle.load(open(file+"tfidf_transformer.pkl", "rb"))
sgdClassifier = pickle.load(open(file+"sgdClassifier.pkl", "rb"))
lstm = load_model(file+"lstm_model.h5")
lstmTokenizer = pickle.load(open(file+"lstmTokenizer.pkl", "rb"))
nltk.download('punkt')


#method to convert feature to pca vectors
def preProcessText(text):
    nonPuncuated = text.translate(str.maketrans('','',string.punctuation))
    STOPWORDS = set(stopwords.words("english"))
    nonStopword = " ".join(word for word in nonPuncuated.split() if word not in STOPWORDS)
    tokenizedText = word_tokenize(nonStopword)
    
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokenizedText)
    lemmatized_words = []

    nonPuncuated = text.translate(str.maketrans('','',string.punctuation))
    STOPWORDS = set(stopwords.words("english"))
    nonStopword = " ".join(word for word in nonPuncuated.split() if word not in STOPWORDS)
    nltk.download('punkt')
    tokenizedText = word_tokenize(nonStopword)
    
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokenizedText)
    lemmatizedWords = []
    for word, tag in pos_tags:
        if tag.startswith('N'):
            pos = 'n'
        elif tag.startswith('V'):
            pos = 'v'
        elif tag.startswith('R'):
            pos = 'r'
        else:
            pos = 'n'
        
        lemma = lemmatizer.lemmatize(word, pos)
        lemmatizedWords.append(lemma)
    joinedText = ' '.join(lemmatizedWords)

    vectorizedText = tfidfVectorizer.transform([joinedText])
    predictedScore = sgdClassifier.predict(vectorizedText)

    tokenizedTextLstm = lstmTokenizer.texts_to_sequences(text)
    paddedText = pad_sequences(tokenizedTextLstm, maxlen=100)
    
    return vectorizedText, paddedText

#method to predict models
def predictingModels(text):
    vectorizedText, paddedText = preProcessText(text)
    predictedScore = sgdClassifier.predict(vectorizedText)
    lstmPrediction = lstm.predict(paddedText)
    max_class = []
    for i in lstmPrediction:
        max_index = np.argmax(i)+1
        max_class.append(max_index)
    lstmValue = max(max_class, key=max_class.count)
    score = (lstmValue+predictedScore[0])//2
    print(score)
    return score

@app.route('/reviewPrediction', methods = ['POST'])
def getText():
    if request.method == "POST":
        data = request.get_json()
        text = data['text']
        print(text)
        if len(text) != 0:
            prediction = predictingModels(text)
            response = jsonify({'score':prediction})
            return response
        else:
            return f"Input doesn't meet prerequisite, Input length is {len(text)}"

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 105)