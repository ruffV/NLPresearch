import re
from transformers import GPT2Tokenizer
import torch


#IMPORTING AND TOKENIZING JOKES.TXT


# Assuming you have a file named "jokes.txt" containing one joke per line
jokes = []
with open("jokes.txt", "r", encoding="utf-8") as file:
    jokes = file.readlines()

# Remove leading/trailing whitespaces and newlines
jokes = [joke.strip() for joke in jokes]

# Remove non-alphanumeric characters and extra spaces
jokes = [re.sub(r"[^a-zA-Z0-9\s]", "", joke) for joke in jokes]

# Remove jokes that are too short or too long (optional)
MIN_JOKE_LENGTH = 5
MAX_JOKE_LENGTH = 100
jokes = [joke for joke in jokes if MIN_JOKE_LENGTH <= len(joke.split()) <= MAX_JOKE_LENGTH]

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Tokenize the jokes
tokenized_jokes = [tokenizer.encode(joke, add_special_tokens=True) for joke in jokes]


#GENERATING X_TRAIN AND Y_TRAIN


# Assume you have tokenized jokes represented as lists of token indices
# For example, tokenized_jokes could look like this:
tokenized_jokes = [
    [1, 4, 7, 9, 2],   # Joke 1, represented as token indices
    [3, 6, 8, 2],      # Joke 2, represented as token indices
    # ...
]

# Add special tokens for the start and end of the sequence (optional, but recommended)
# In this example, we use 0 as the index for the start token and -1 for the end token
# This assumes that your vocabulary starts with index 1 (0 is reserved for padding)
for joke_tokens in tokenized_jokes:
    joke_tokens.insert(0, 0)    # Add start token (0) at the beginning of each joke
    joke_tokens.append(-1)      # Add end token (-1) at the end of each joke

# Convert tokenized_jokes into PyTorch tensors
X_train = torch.tensor([joke_tokens[:-1] for joke_tokens in tokenized_jokes], dtype=torch.long)
y_train = torch.tensor([joke_tokens[1:] for joke_tokens in tokenized_jokes], dtype=torch.long)


