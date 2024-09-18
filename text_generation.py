import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from collections import Counter
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import matplotlib.pyplot as plt

import sys
import os

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Read command-line arguments
source_text = sys.argv[1]
seed_text = sys.argv[2]
epochs = sys.argv[3]

# Read and preprocess data
data_path = os.path.join("./cleaned_source_text/", source_text)
with open(data_path, 'r', encoding='utf-8') as f:
    data = f.read()

corpus = data.lower().split("\n")

# Tokenize and build vocabulary
def tokenize_corpus(corpus):
    tokenizer = {}
    word_counts = Counter()
    for line in corpus:
        tokens = line.strip().split()
        word_counts.update(tokens)
    word_index = {word: idx + 1 for idx, word in enumerate(word_counts)}
    index_word = {idx: word for word, idx in word_index.items()}
    total_words = len(word_index) + 1  # +1 for padding index 0
    return word_index, index_word, total_words

word_index, index_word, total_words = tokenize_corpus(corpus)

# Create input sequences and labels
def create_sequences(corpus, word_index):
    input_sequences = []
    for line in corpus:
        token_list = [word_index[word] for word in line.strip().split() if word in word_index]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

input_sequences = create_sequences(corpus, word_index)

# Pad sequences
max_sequence_len = max(len(seq) for seq in input_sequences)

def pad_sequences(sequences, max_len, padding_value=0):
    padded = []
    for seq in sequences:
        seq = torch.tensor(seq, dtype=torch.long)
        if len(seq) < max_len:
            seq = torch.cat([torch.full((max_len - len(seq),), padding_value, dtype=torch.long), seq])
        else:
            seq = seq[-max_len:]
        padded.append(seq)
    return torch.stack(padded)

padded_sequences = pad_sequences(input_sequences, max_sequence_len, padding_value=0)

# Split predictors and labels
predictors = padded_sequences[:, :-1]
labels = padded_sequences[:, -1]

# Create Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, predictors, labels):
        self.predictors = predictors
        self.labels = labels

    def __len__(self):
        return len(self.predictors)

    def __getitem__(self, idx):
        return self.predictors[idx], self.labels[idx]

dataset = TextDataset(predictors, labels)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the model
class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim1, hidden_dim2, dropout):
        super(TextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim1, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1 * 2, hidden_dim2, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim2, vocab_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(vocab_size // 2, vocab_size)
    
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out1, _ = self.lstm1(embeds)
        lstm_out1 = self.dropout(lstm_out1)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = lstm_out2[:, -1, :]  # Output from the last time step
        out = self.fc1(lstm_out2)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Instantiate the model
model = TextGenerationModel(
    vocab_size=total_words,
    embedding_dim=100,
    hidden_dim1=150,
    hidden_dim2=100,
    dropout=0.2
).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = epochs
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch_inputs, batch_labels in dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)

        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    
    avg_loss = epoch_loss / len(dataset)
    accuracy = correct / total

    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%')

# Plot training accuracy and loss
plt.plot(range(len(train_accuracies)), train_accuracies, 'b', label='Training Accuracy')
plt.title("Training Accuracy")
plt.legend()
plt.figure()

plt.plot(range(len(train_losses)), train_losses, 'b', label="Training Loss")
plt.title("Training Loss")
plt.legend()
# plt.show()

# Text generation
model.eval()
next_words = 100

for _ in range(next_words):
    token_list = [word_index.get(word, 0) for word in seed_text.strip().split()]
    token_list = token_list[-(max_sequence_len - 1):]  # Limit to max_sequence_len - 1
    token_list = [0] * (max_sequence_len - 1 - len(token_list)) + token_list  # Pad with zeros

    token_tensor = torch.tensor(token_list, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(token_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
    
    output_word = index_word.get(predicted_index, '')
    seed_text += ' ' + output_word

print("Generated text:")
print(seed_text)
