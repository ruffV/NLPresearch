import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
nltk.download('punkt')  # Download the Punkt tokenizer models (only needed once)

def read_jokes_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        jokes = file.readlines()
    return jokes

def tokenize_jokes(tokenizer, jokes):
    tokenized_jokes = [tokenizer.encode(joke.strip(), add_special_tokens=True) for joke in jokes]
    return tokenized_jokes

def setup_gpt2_model():
    # Load the pre-trained GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load the pre-trained GPT-2 model (weights)
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Set the model to training mode (important for fine-tuning)
    model.train()

    return tokenizer, model


jokes_file_path = 'testjokes.txt'  # Replace with the path to your jokes file
jokes = read_jokes_from_file(jokes_file_path)

# Setup GPT-2 model and tokenizer
tokenizer, model = setup_gpt2_model()

# Tokenize the jokes using the GPT-2 tokenizer
tokenized_jokes = tokenize_jokes(tokenizer, jokes)

# Print the tokenized jokes
#for i, joke_tokens in enumerate(tokenized_jokes, 1):
#    print(f"Joke {i}: {joke_tokens}")





max_sequence_length = 64  # Set the maximum sequence length

input_ids = [torch.tensor(joke[:max_sequence_length]) for joke in tokenized_jokes]

# Pad sequences to the same length
padding_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0.0
input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=padding_value)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_ids = input_ids.to(device)
model.to(device)


# Create DataLoader for batching the data
batch_size = 8  # Set your desired batch size I REALLY DONT KNOW
train_dataset = TensorDataset(input_ids)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss()
YOUR_LEARNING_RATE = 1 * (10 ** -4)
optimizer = torch.optim.Adam(model.parameters(), lr=YOUR_LEARNING_RATE)


YOUR_NUMBER_OF_EPOCHS = 10 # I REAlly Dont KNOW
# Training loop
for epoch in range(YOUR_NUMBER_OF_EPOCHS):
    model.train()  # Set the model to training mode

    for batch_inputs in train_loader:
        optimizer.zero_grad()  # Clear gradients for each batch

        # Move batch data to the device (e.g., GPU)
        batch_inputs = batch_inputs[0].to(device)

        # Forward pass
        outputs = model(batch_inputs, labels=batch_inputs)

        # Calculate the loss
        loss = outputs.loss

        # Backpropagation
        loss.backward()
        optimizer.step()

    # Print the loss for each epoch
    print(f"Epoch [{epoch+1}/{YOUR_NUMBER_OF_EPOCHS}], Loss: {loss.item()}")

# Save the trained model (optional)
torch.save(model.state_dict(), "gpt2_joke_generator.pth")