# Copied from https://github.com/huggingface/transformers/tree/main/examples/research_projects/codeparrot

from datasets import load_dataset
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoTokenizer, HfArgumentParser
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


@dataclass
class TokenizerTrainingArguments:
    """
    Configuration for tokenizer training.
    """

    base_tokenizer: Optional[str] = field(
        default="bigcode/starcoder",
        metadata={"help": "Base tokenizer to build new tokenizer from."},
    )
    dataset_name: Optional[str] = field(
        default="smangrul/hug_stack",
        metadata={"help": "Dataset to train tokenizer on."},
    )
    text_column: Optional[str] = field(
        default="text", metadata={"help": "Column containing text data to process."}
    )
    vocab_size: Optional[int] = field(
        default=50_000, metadata={"help": "Number of examples to train tokenizer on."}
    )
    n_examples: Optional[int] = field(
        default=6500,
        metadata={"help": "Number of examples to train the tokenizer on."},
    )
    tokenizer_name: Optional[str] = field(
        default="hugcoder", metadata={"help": "Name of new tokenizer."}
    )
    push_to_hub: Optional[bool] = field(
        default=True, metadata={"help": "Push saved tokenizer to the hub."}
    )


def main():
    # This function will create and train a new tokenizer using a dataset.
    # A tokenizer is a tool that splits text into smaller pieces (tokens), which are used as input for language models.
    # Training a tokenizer on your own data helps the model better understand the specific language or code in your dataset.

    # Iterator for Training
    def batch_iterator(batch_size=10):
        # This function generates batches of text data for training the tokenizer.
        # Instead of loading all data into memory at once (which can be very large),
        # it loads a small batch at a time, making the process memory-efficient.
        #
        # Each batch is a list of text samples from the dataset.
        # The tqdm progress bar shows how many batches have been processed so far.
        for _ in tqdm(range(0, args.n_examples, batch_size)):
            # For each batch, get 'batch_size' number of text samples from the dataset.
            # 'next(iter_dataset)' gets the next example from the dataset iterator.
            # '[args.text_column]' extracts the text field from each example.
            yield [next(iter_dataset)[args.text_column] for _ in range(batch_size)]

    # Parse command-line arguments to get configuration for tokenizer training.
    # This allows you to easily change things like which dataset to use, how many examples, etc.
    parser = HfArgumentParser(TokenizerTrainingArguments)
    args = parser.parse_args()

    # Load the base tokenizer from HuggingFace.
    # This is a pre-existing tokenizer that we will use as a starting point.
    # Training a new tokenizer from scratch is possible, but starting from a good base often gives better results.
    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)

    # Get the base vocabulary (alphabet) for the tokenizer.
    # This ensures that all basic characters (like letters, numbers, punctuation) are included in the new tokenizer.
    base_vocab = list(bytes_to_unicode().values())

    # Load the dataset for training the tokenizer.
    # 'streaming=True' means we don't load the entire dataset into memory, but read it sample by sample.
    # 'split="train"' selects the training portion of the dataset.
    dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    # Create an iterator so we can read the dataset one example at a time.
    iter_dataset = iter(dataset)

    # Train a new tokenizer using the text data from the dataset.
    # 'train_new_from_iterator' takes batches of text and learns how to split them into tokens.
    # 'vocab_size' controls how many unique tokens the tokenizer will have.
    # 'initial_alphabet' ensures all basic characters are included.
    new_tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(), vocab_size=args.vocab_size, initial_alphabet=base_vocab
    )

    # Save the trained tokenizer to disk, and optionally upload it to the HuggingFace Hub for sharing.
    new_tokenizer.save_pretrained(args.tokenizer_name, push_to_hub=args.push_to_hub)


if __name__ == "__main__":
    main()
