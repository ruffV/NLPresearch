import torch
import torch.nn.functional as F

# Load the trained model (make sure to replace 'joke_generator_model.pth' with your model's filename)
model = JokeGenerator(vocab_size, embedding_dim, hidden_size)
model.load_state_dict(torch.load('joke_generator_model.pth'))
model.eval()  # Set the model to evaluation mode

# Start text generation with a given seed text
seed_text = "Why did the chicken cross the road?"
generated_text = seed_text

# Set the number of tokens to generate
num_tokens_to_generate = 50

# Convert seed text to tokens
tokens = [token_to_index[token] for token in seed_text.split()]

# Generate text
with torch.no_grad():
    for i in range(num_tokens_to_generate):
        # Convert tokens to a PyTorch tensor and move to the appropriate device (e.g., GPU)
        input_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

        # Forward pass to get the model's output distribution for the next token
        outputs = model(input_tensor)

        # Retrieve the last word's output distribution (next token distribution)
        next_token_probs = outputs[:, -1, :]

        # Sample the next token probabilistically
        next_token_idx = torch.multinomial(F.softmax(next_token_probs, dim=-1), num_samples=1)

        # Convert the sampled token index to its corresponding word
        next_token = index_to_token[next_token_idx.item()]

        # Append the sampled token to the generated text
        generated_text += " " + next_token

        # Update the input tokens list for the next iteration
        tokens.append(next_token_idx.item())

# Print the generated text
print(generated_text)