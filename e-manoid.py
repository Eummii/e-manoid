import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

try:
    model = tf.keras.models.load_model('e-manoid.h5')
except Exception as e:
    print("Error loading the model. Please make sure the model file 'e-manoid.h5' is available.")
    exit(1)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_prob = max(res)
    threshold = max_prob * 0.8  # Set threshold to 80% of the maximum probability
    results = [{'intent': classes[i], 'probability': str(r)} for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x['probability'], reverse=True)
    return results

# Test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    predictions = predict_class(user_input)
    print("E-manoid:")
    for pred in predictions:
        intent_tag = pred['intent']
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                responses = intent['responses']
                response = random.choice(responses)
                print(f"- {response} (Probability: {pred['probability']})")
