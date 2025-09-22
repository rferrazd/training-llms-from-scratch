
---
license: apache-2.0
tags:
- tokenizer
- bpe
- wikitext2
- nlp
---

# WikiText-2 BPE Tokenizer

A Byte Pair Encoding (BPE) tokenizer trained on the WikiText-2 dataset.

## Model Details
- **Vocabulary Size**: 30,000 tokens
- **Training Data**: WikiText-2 (Salesforce/wikitext)
- **Special Tokens**: [PAD], [UNK], [CLS], [SEP], [MASK]
- **Compression Ratio**: ~6.4 characters per token

## Usage
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Rogarcia18/wikitext2-bpe-tokenizer")
```

## Training Details
- Dataset: WikiText-2 (wikitext-2-v1)
- Preprocessing: Deduplication, <unk> removal, whitespace normalization, remove samples cases with less than 10 characters
- Architecture: BPE with HuggingFace tokenizers library
