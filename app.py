from flask import Flask, jsonify, render_template, request
import google.generativeai as genai
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

# Load prediction model
stemmer = LancasterStemmer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents2.json').read())
words = pickle.load(open('words.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    return return_list


# Load Gemini AI model
genai.configure(api_key='AIzaSyAg6UtggTP8rYwWQ-oBhJQf7xDa7SyyhpE')
gemini_model = genai.GenerativeModel('gemini-pro')
chat = gemini_model.start_chat(history=[])

app = Flask(__name__)
chat_history = []


@app.route('/')
def index():
    return render_template('chat.html', chat_history=chat_history)


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        user_input = request.json.get('user_input')

        if user_input:
            # Get response from prediction model
            prediction_response = predict_class(user_input)
            # Get response from Gemini AI model
            gemini_response = chat.send_message(user_input)
    
            # Update chat history with both responses
            chat_history.append({"user": user_input,
                                 "prediction_bot": prediction_response,
                                 "gemini_bot": gemini_response.text})

            print("User Input:", user_input)
            print("Prediction Response:", prediction_response)
            print("Gemini Response:", gemini_response.text)

            return jsonify({
                "prediction_response": prediction_response,
                "gemini_response":  gemini_response.text
            })
        else:
            return jsonify({"error": "No user input provided."}), 400
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
