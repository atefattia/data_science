import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re # use regular expression
import string

app = Flask(__name__)
model = load_model('./saved_model')

stop = set(stopwords.words('german'))
def remove_stop_words(data):
    for i, sentence in enumerate(data):
        sentence_without_stops = [i for i in word_tokenize(sentence[0].lower()) 
                                  if i not in stop]
        sent = [TreebankWordDetokenizer().detokenize(sentence_without_stops)]
        data[i] = sent
    return data

def clean_text(text):
    text = text.lower() # lowercase 
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # ponctuation
    text = re.sub('\d', '', text) # numbers
    text = re.sub('[''–’-“”_]', '', text) # special character
    text = re.sub('\n', '', text) # new line
    return text

map_one_hot_label = {0: 'Wirtschaft',
                     1: 'Web',
                     2: 'Kultur',
                     3: 'Gesundheit',
                     4: 'Sport',
                     5: 'Wissenschaft'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    raw_text = request.form.get('myText')
    text_list = [request.form.get('myText')]
    text_clean = np.array([clean_text(x) for x in text_list])
    text_wo_stop_words = remove_stop_words(text_clean.reshape(-1, 1))
    prediction = model.predict(text_wo_stop_words)
    output = map_one_hot_label[np.argmax(prediction, axis=1)[0]]

    return render_template('index.html',
        raw_text=raw_text,
        prediction_text='The article\'s category is: -{}-'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=8080)