import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

from sys import argv

# Check for GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable dynamic memory allocation
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print('Running on multiple GPUs')
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            print('Running on a single GPU')
    except RuntimeError as e:
        print(e)
else:
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    print("No GPUs found. Running on CPU.")

source_text = argv[1]

# Text cleaning
tokenizer = Tokenizer()
data = open("./cleaned_source_text/" + source_text).read()
corpus = data.lower().split("\n")

# Tokenize words
tokenizer.fit_on_texts(corpus)

# Remember total number of words
total_words = len(tokenizer.word_index) + 1

# Create input sequences using the list of tokens
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i + 1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
)

# Create predictors and label
predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
label = ku.to_categorical(label, num_classes=total_words)

with strategy.scope():
    # Build the model
    model = Sequential()
    model.add(
        Embedding(
            total_words,
            100,
            input_length=max_sequence_len - 1,
        )
    )
    model.add(Bidirectional(LSTM(150, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(
        Dense(
            total_words // 2,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01),
        )
    )
    model.add(Dense(total_words, activation='softmax'))

    # Compile the model
    model.compile(
        loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
    )

# Implement early stopping
early_stop = EarlyStopping(
    monitor='loss',
    mode='min',
    verbose=1,
    patience=10,
    restore_best_weights=True,
)

# Train the model
results = model.fit(
    predictors,
    label,
    epochs=200,
    callbacks=[early_stop],
    verbose=1,
)

# Plot training accuracy and loss
acc = results.history['accuracy']
loss = results.history['loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.title("Training Accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label="Training Loss")
plt.title("Training Loss")
plt.legend()
# plt.show()

# Generate text
seed_text = argv[2]
next_words = 100

for _ in range(next_words):
    # Tokenize the seed sentence
    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    # Pad the token list
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_len - 1, padding='pre'
    )

    # Predict the next word
    predictions = model.predict(token_list, verbose=0)
    most_probable = np.argmax(predictions, axis=1)

    output_word = ""

    for word, index in tokenizer.word_index.items():
        if index == most_probable:
            output_word = word
            break

    seed_text += " " + output_word

print("Generated text:")
print(seed_text)