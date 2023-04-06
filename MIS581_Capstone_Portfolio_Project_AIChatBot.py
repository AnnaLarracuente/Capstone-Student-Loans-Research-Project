#!/usr/bin/env python
# coding: utf-8

# # An Artificial Intelligence Tri-Lingual ChatBot Solution for Identifying Which Americans Are at Risk for Defaulting on Their Student Loans (2023)
# 
# ##Anna Larracuente

# In[1]:


pip install tensorflow


# In[2]:


# Import Necessary Libraries & Tools

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
model = load_model('chatbot_model.h5')
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

import json
import random
import pickle

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import tkinter
from tkinter import *


# In[3]:


# Initialize Chatbot Training

words = []
classes = []
documents = []
ignore_words = ['.','!','?',':',',',';']

# Read Custom 'student_loans_intents' JSON File & Assign to Object 'intents'

data_file = open('student_loans_intents.json').read()
intents = json.loads(data_file)


# In[4]:


# Extract Words From Patterns (From JSON Data File) Via Nested Loop

for intent in intents['intents']:
    for pattern in intent['patterns']:
    
        #Tokenize Each Word
        
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        
        #Add to Documents List Each Pair of Patterns W/ Corresponding Tags
        
        documents.append((w, intent['tag']))

        #Add Corresponding Tags to Classes List, Prevent Repeats
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# In[5]:


# Lemmatize All Words and Sort 'words' & 'classes' Lists

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))


# In[6]:


#Initialize Training Data with "training" Variable

training = []
output_empty = [0] * len(classes)

#Create Nested List Containing Bag of Words (BOW) for Each Document

for doc in documents:
    
    #Initialize BOW
    
    bag = []
    
    #List of Tokenized words for Pattern
    
    pattern_words = doc[0]
    
    #Lemmatize Each Word in Pattern_Words List
    
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    #Create BOW Array with 1 (if Word Match is Found in Current Pattern)
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

#Output_Row Serves as a Key for the List. Output = '0' for Each Tag & Output = '1' for Current Tag (for Each Pattern)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

#Append BOW & Output_Row "Keys"
    
    training.append([bag, output_row])

#Shuffle Training Set and Convert into NumPy Array

random.shuffle(training)
training = np.array(training)

#Perform Manual Train/Test Split (Patterns = X & Intents = Y)  

train_x = list(training[:,0])
train_y = list(training[:,1])


# In[7]:


#Create a Multi-Layer Perceptron Neural Network (NN) Model with 3 layers:
#First Layer = 300 Neurons
##Second Layer = 100 Neurons
###3rd (Output Layer) = # of Neurons = # of Intents

#Leverage Sequential Keras DL Model to Build NN, Predict Output Intent with 'softmax'

model = Sequential()
model.add(Dense(300, input_shape = (len(train_x[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation ='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax'))

#Compile NN Model: Stochastic Gradient Descent with Nesterov Accelerated Gradient

sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#Assigning Fitted Model to "hist" After Conversion to NumPy Array

hist = model.fit(np.array(train_x), np.array(train_y), epochs = 1000, batch_size = 5, verbose = 1)

#Save Trained Model as "chatbot_model.h5"

model.save('chatbot_model.h5', hist)


# In[8]:


#Setting Up Chatbot to be Later Accessed Via Custom GUI

#Define Function to "Clean Up" Inputted Sentences

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

#Define Function "bow" as Complete Bag of Words + Cleaned Up Sentences Used for Predicting Classes (0 or 1 for Each Word in BOW that Exists in Sentence)

def bow(sentence, words, show_details = True):
    
    #Tokenize & Lemmatize Words in Input Sentences ("Clean Up")
    
    sentence_words = clean_up_sentence(sentence)
    
    #BOW - Matrix of N Words, Vocabulary Matrix
    
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                
                # = 1 if Current Word is in the Vocabulary Position
                
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    
    #Return BOW as NumPy Array
    
    return(np.array(bag))

#Define Function to Output a List of Intents & Respective Probabilities of Matching the Correct Intent

def predict_class(sentence, model):
    
    #Filter Out Predictions Below the Error Threshold (0.25, to Avoid Excessive Overfitting)
    
    p = bow(sentence, words,show_details = False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    #Sort by Strength of Probability
    
    results.sort(key = lambda x: x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

#Define Function to Take the Outputted List, Check the JSON file, & Output a Response with the Highest Probability

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
    return result

#Define Function to Take an Input Message, 'Predict the Class' & 'Get a Response', then Outputs a Chatbot Response

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


# In[9]:


#Create a Custom Chatbot Graphical User Interface (GUI) with 'tkinter'

#Define Function to Set Up Basic Functionality of Chatbot 

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    #If Input Message is Not an Empty String, the Chatbot Will Output a Response Based on the 'chatbot_response()' Function
    
    if msg != '':
        ChatLog.config(state = NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground = "#442265", font = ("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Chatbot Alex: " + res + '\n\n')

        ChatLog.config(state = DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("Chat with Alex")
base.geometry("400x500")
base.resizable(width = FALSE, height = FALSE)

#Build Chat Window
ChatLog = Text(base, bd = 0, bg = "#9cffe4", height = "8", width = "50", font = "Verdana",)

ChatLog.config(state = DISABLED)

#Build and Bind Scrollbar to Chat Window

scrollbar = Scrollbar(base, command = ChatLog.yview, cursor = "heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create a Button to Send (Input) Messages

SendButton = Button(base, font = ("Verdana", 12, 'bold'), text = "Send", width = "12", height = 5,
                    bd = 0, bg = "#32de97", activebackground = "#3c9d9b", fg = "#000000",
                    command = send)

#Create the Text Box to Enter User Messages

EntryBox = Text(base, bd = 0, bg = "#ffffff", width = "29", height = "5", font = "Verdana")
#EntryBox.bind("<Return>", send)


#Place All Combined Components onto the Screen, Specifying Coordinates & Heights

scrollbar.place(x = 376, y = 6, height = 386)
ChatLog.place(x = 6, y = 6, height = 386, width = 370)
EntryBox.place(x = 128, y = 401, height = 90, width = 265)
SendButton.place(x = 6, y = 401, height = 90)

base.mainloop()

