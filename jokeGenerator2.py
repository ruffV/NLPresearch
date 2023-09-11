import torch
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
# Load the model architecture
model = GPT2LMHeadModel.from_pretrained('gpt2')  # Load the GPT-2 model
model.load_state_dict(torch.load('gpt2_joke_generator.pth'))
model.eval()  # Set the model to evaluation mode

# Prepare input prompt (text seed)
input_prompt = "Why did the chicken cross the road?"


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Tokenize the input (you should use the same tokenizer used during training)
#input_tokens = tokenize(input_prompt)  # Replace with your tokenization code
input_tokens = tokenizer.encode(input_prompt, add_special_tokens=True)

# Maximum length of the generated text
max_length = 50  # You can adjust this as needed

# Generate text
generated_text = input_tokens  # Start with the input tokens

with torch.no_grad():
    while len(generated_text) < max_length:
        input_ids = torch.tensor(generated_text).unsqueeze(0)  # Convert to tensor
        outputs = model(input_ids)  # Make predictions
        logits = outputs.logits
        predicted_token_id = torch.argmax(logits, dim=-1)[:, -1].item()  # Get the last predicted token
        generated_text.append(predicted_token_id)

# Convert token IDs back to text (you should use your tokenizer's decode function)
#generated_text = decode(generated_text)  # Replace with your decoding function
generated_text = tokenizer.decode(generated_text, skip_special_tokens=True)

print("Generated Joke: ", generated_text)