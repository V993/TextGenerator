# RNNs have dependency on context only from the past
# Bidirectional LSTMs can be used to train two sides, 
# instead of one side of the input sequence. This results
# in faster learning, as well as more context to the word
# which is conducive to a more full understanding of the
# word.

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

from sys import argv

source_text = argv[1]

# text cleaning
tokenizer = Tokenizer()
data = open("./cleaned_source_text/"+source_text).read()
corpus = data.lower().split("\n")

# tokenize words
tokenizer.fit_on_texts(corpus)

# remember total number of words
total_words = len(tokenizer.word_index) +1


# create input sequences using list of tokens
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# padding sequences
max_sequence_len = max(
                        [len(x) for x in input_sequences]
                      )

input_sequences = np.array(pad_sequences(input_sequences,
                                    maxlen=max_sequence_len, padding='pre'))
    
# extract the alst word of the sequence and convert from numerical to categorical

# create predictors and label
predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
label = ku.to_categorical(label, num_classes=total_words)

# now we make a sequential model:
model = Sequential()

# we make the first layer an embedding layer 
model.add(
    Embedding(total_words, 100,
                    input_length=max_sequence_len-1)
)

# the second layer will be a bidirectional LSTM
model.add(
    Bidirectional(LSTM(150, return_sequences = True))
        # return sequence flag True so that the word generation keeps in consideration previous
        # and future words in the sequence
)

# dropout layer to avoid overfitting
model.add(
    Dropout(0.2)
)

# another LSTM layer
model.add(
    LSTM(100)
)

# add a dense layer with ReLU activation plus a regularizer layer to avoid overfitting
model.add(
    Dense(total_words/2, activation='relu',
           kernel_regularizer=regularizers.l2(0.01))
)

# softmax to get the probability of the word to be predicted next
model.add(
    Dense(total_words, activation='softmax')
)

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# output model
# print(model.summary())

#implement early stopping
early_stop = EarlyStopping( monitor='val_loss', mode='min', 
                            verbose=1, patience=10, 
                            min_delta=10, restore_best_weights=True)

results = model.fit(predictors, 
                    label, 
                    epochs=200, 
                    callbacks = [early_stop],
                    verbose=1)

acc = results.history['accuracy']
loss = results.history['loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.title("Training Accuracy")
plt.figure()

plt.plot(epochs, loss, 'b', label="Training Loss")
plt.title("Training Loss")
plt.legend()
# plt.show()

seed_text = "Individual Americans have a right to own firearms."
next_words = 100

for _ in range(next_words):
    # tokenize the seed sentence
    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    # pad onto the token list
    token_list = pad_sequences([token_list],
        maxlen=max_sequence_len-1, padding='pre')

    # predict from seed
    predictions = model.predict(token_list, verbose=1)
    most_probable = np.argmax(predictions,axis=1)

    output_word = "sike"

    print(output_word,most_probable)

    for word, index in tokenizer.word_index.items():

        if index == most_probable: 
            output_word = word
            break

    seed_text += " " + output_word

print(seed_text)