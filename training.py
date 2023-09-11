#OLD

import re
from transformers import GPT2Tokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
PAD_INDEX = 5

#IMPORTING AND TOKENIZING JOKES.TXT
# Assuming you have a file named "jokes.txt" containing one joke per line
jokes = []
with open("testJokes.txt", "r", encoding="utf-8") as file:
    jokes = file.readlines()

# Remove leading/trailing whitespaces and newlines
jokes = [joke.lower().strip() for joke in jokes]

# Remove non-alphanumeric characters and extra spaces
jokes = [re.sub(r"[^a-zA-Z0-9\s]", "", joke) for joke in jokes]

# Remove jokes that are too short or too long (optional)
MIN_JOKE_LENGTH = 5
MAX_JOKE_LENGTH = 100
jokes = [joke for joke in jokes if MIN_JOKE_LENGTH <= len(joke.split()) <= MAX_JOKE_LENGTH]

print(jokes)
print("\n"*10)




# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Tokenize the jokes
tokenized_jokes = [tokenizer.encode(joke, add_special_tokens=True) for joke in jokes]

print(tokenized_jokes)
print("\n"*10)

#GENERATING X_TRAIN AND Y_TRAIN
# Assume you have tokenized jokes represented as lists of token indices

# Add special tokens for the start and end of the sequence (optional, but recommended)
# In this example, we use 0 as the index for the start token and -1 for the end token
# This assumes that your vocabulary starts with index 1 (0 is reserved for padding)


for joke_tokens in tokenized_jokes:
    joke_tokens.insert(0, 0)    # Add start token (0) at the beginning of each joke
    joke_tokens.append(-1)      # Add end token (-1) at the end of each joke

print(tokenized_jokes)


# Create a vocabulary (set of unique tokens) from the tokenized jokes
vocabulary = set(token for joke_tokens in tokenized_jokes for token in joke_tokens)

# Create token-to-index mapping (token_to_index)
token_to_index = {token: index for index, token in enumerate(vocabulary)}
# Convert tokenized jokes to PyTorch tensors
joke_tensors = [torch.tensor([token_to_index[token] for token in joke_tokens], dtype=torch.long) for joke_tokens in tokenized_jokes]

# Pad the sequences to a fixed length (max_length)
max_length = max(len(joke) for joke in joke_tensors)
X_train = pad_sequence([joke[:-1] for joke in joke_tensors], batch_first=True, padding_value=PAD_INDEX)
y_train = pad_sequence([joke[1:] for joke in joke_tensors], batch_first=True, padding_value=PAD_INDEX)

# Convert tokenized_jokes into PyTorch tensors

#X_train = torch.tensor([joke_tokens[:-1] for joke_tokens in tokenized_jokes], dtype=torch.long)
#y_train = torch.tensor([joke_tokens[1:] for joke_tokens in tokenized_jokes], dtype=torch.long)



#MODEL CREATION

vocab_size = len(tokenized_jokes)
embedding_dim = 50

# Create the language model
class JokeGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(JokeGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        output = self.fc(output)
        return output

# Initialize the model
hidden_size = 256  # You can experiment with the hidden size
model = JokeGenerator(vocab_size, embedding_dim, hidden_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Print the model summary
print(model)




#MODEL TRAINING
batch_size = 32#still confused what this means
num_epochs = 10

# Create DataLoader for batching and shuffling your data
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Check if GPU is available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()  # Clear gradients for each batch

        # Move batch data to the device (e.g., GPU)
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        # Forward pass
        outputs = model(batch_inputs)

        # Calculate the loss
        loss = criterion(outputs.view(-1, vocab_size), batch_targets.view(-1))

        # Backpropagation
        loss.backward()
        optimizer.step()

    # Print the loss for each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Save the trained model (optional)
torch.save(model.state_dict(), "joke_generator_model.pth")