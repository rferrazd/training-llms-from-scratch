
from transformers import AutoTokenizer

# Load the tokenizer from local HuggingFace format files
tokenizer = AutoTokenizer.from_pretrained("my_bpe_tokenizer_hf")

# --- OR --- 
# Load the tokenizer from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained("Rogarcia18/wikitext2-bpe-tokenizer")
# ----------


# Examples of how to use the tokenizer:

# ---------- Example 1 ----------
text1 = "Hello world! This is a simple example."
print("=" * 60)
print("EXAMPLE 1")
print("=" * 60)
print(f"Original text: '{text1}'")

# Encode the text
encoded1 = tokenizer.encode(text1)
tokens1 = tokenizer.convert_ids_to_tokens(encoded1)
print(f"Tokens: {tokens1}")
print(f"Token IDs: {encoded1}")
print(f"Number of tokens: {len(encoded1)}")

# Decode back to text
decoded1 = tokenizer.decode(encoded1)
print(f"Decoded text: '{decoded1}'")
print()

# ---------- Example 2 ----------
text2 = "Hello\n how will my tokenizer tokenize the word: `blahblah`?."
print("=" * 60)
print("EXAMPLE 2")
print("=" * 60)
print(f"Original text: '{text2}'")

encoded2 = tokenizer.encode(text2)
tokens2 = tokenizer.convert_ids_to_tokens(encoded2)
print(f"Tokens: {tokens2}")
print(f"Token IDs: {encoded2}")
print(f"Number of tokens: {len(encoded2)}")

decoded2 = tokenizer.decode(encoded2)
print(f"Decoded text: '{decoded2}'")
print()

# ---------- Example 3 ----------
text3 = "This is my first tokenizer! I pushed it to my repo: Rogarcia18/wikitext2-bpe-tokenizer"
print("=" * 60)
print("EXAMPLE 3")
print("=" * 60)
print(f"Original text: '{text3}'")

encoded3 = tokenizer.encode(text3)
tokens3 = tokenizer.convert_ids_to_tokens(encoded3)
print(f"Tokens: {tokens3}")
print(f"Token IDs: {encoded3}")
print(f"Number of tokens: {len(encoded3)}")
decoded3 = tokenizer.decode(encoded3)
print(f"Decoded text: '{decoded3}'")


