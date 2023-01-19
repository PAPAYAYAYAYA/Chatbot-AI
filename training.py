import random
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

# Load intents
with open('intents.json') as json_file:
    intents = json.load(json_file)

# Extract patterns and labels
patterns, labels = [], []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        labels.append(intent['tag'])

# Tokenize the patterns
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)

# Get the word index
word_index = tokenizer.word_index

# Get the maximum sequence length
max_sequence_length = max([len(sequence) for sequence in sequences])

# Pad the sequences
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2)

# Create categorical labels
labels = np.asarray(labels)
label_index = {label: i for i, label in enumerate(np.unique(labels))}
labels = np.array([label_index[label] for label in labels])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2)

# Define the model
model = Sequential()
model.add(Embedding(len(word_index)+1, 128, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(len(label_index), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(sequences, labels, validation_split=0.2, epochs=200, batch_size=5, verbose=1)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('chatbot_model.h5')

print("Model training and saving complete!")

