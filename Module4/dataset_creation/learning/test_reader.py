
# File to test the reader with our dummy files
import sys
import os
# Add the parent directory (dataset_creation) to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
# Now import from reader module
from reader import PersonalCopilotDatasetReader

"""
1. The PersonalCopilotDatasetReader constructor requires a DataFolderLike object as its first parameter (not a simple string path)

2. DataFolder provides an abstraction for file operations that works with different storage backends

3. It handles path normalization and provides methods like open() that the reader uses to read files

"""

# TESTING FUNCTION
def test_reader():

    # Create a DataFolder pointing to the desired directory: dummy_docs
    data_folder = os.path.join(os.path.dirname(__file__), "dummy_docs")

    # Initialize the PersonalCopilotDatasetReader with that folder
    reader = PersonalCopilotDatasetReader(data_folder = data_folder)

    # Process the file(s)
    for doc in reader.run():
        content = doc.text
        metadata = doc.metadata
        # Print the content and metadata extracted
        print(f"File: {metadata['file_path']}")
        print(f"Content: {content}...")  # Print first 100 chars
        print("-" * 50)

if __name__ == "__main__":
    test_reader()
    