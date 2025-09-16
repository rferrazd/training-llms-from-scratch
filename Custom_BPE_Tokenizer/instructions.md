# Problem Statement: Building a Custom BPE Tokenizer Using WikiText-2 Dataset

You are tasked with building a custom Byte Pair Encoding (BPE) tokenizer from scratch using the WikiText-2 dataset, available on Hugging Face at Salesforce/wikitext ([https://huggingface.co/datasets/Salesforce/wikitext]). The tokenizer should be optimized for modeling English text and usable in downstream language modeling tasks.

Objectives:
Load and Explore the Dataset:


Use the wikitext dataset from Hugging Face's datasets library.


Select the WikiText-2 subset ("wikitext-2-v1"). This will contain 44.8k rows.


Data Cleaning & Preprocessing:


Perform deduplication of the training text to eliminate exact duplicates.


Normalize and clean the text if necessary (e.g., remove special tokens like <unk> or unnecessary whitespace).


Tokenizer Training:


Implement or use Hugging Face’s tokenizers library to build a BPE tokenizer from scratch.


Define and train the tokenizer on the deduplicated training set.


Specify parameters like:


Vocabulary size (e.g., 30,000 tokens)


Special tokens: [PAD], [UNK], [CLS], [SEP], [MASK]


Tokenizer Evaluation:


Test the trained tokenizer on the validation and test splits of WikiText-2.


Report metrics such as:


Vocabulary size


Tokenization consistency


Average tokens per sentence


Compression ratio


Save and Reuse:


Save the tokenizer in Hugging Face-compatible format.


Demonstrate reloading and using the tokenizer for encoding/decoding text.