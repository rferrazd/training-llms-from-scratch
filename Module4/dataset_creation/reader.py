# coding=utf-8
# Copyright 2024 Sourab Mangrulkar. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module provides a custom dataset reader for processing code and notebook files,
with filtering for unwanted file types and special handling for Jupyter notebooks.
It includes utilities for segmenting notebook cells, cleaning markdown, and formatting
the extracted content for downstream processing.
"""

import itertools
import json
import re
from typing import Callable

from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.io import DataFolderLike

# Block the following formats.
IMAGE = ["png", "jpg", "jpeg", "gif"]
VIDEO = ["mp4", "jfif"]
DOC = ["key", "PDF", "pdf", "docx", "xlsx", "pptx", "csv", "tsv", "txt"]
AUDIO = ["flac", "ogg", "mid", "webm", "wav", "mp3"]
ARCHIVE = ["jar", "aar", "gz", "zip", "bz2"]
MODEL = ["onnx", "pickle", "model", "neuron"]
OTHERS = [
    "npy",
    "index",
    "inv",
    "index",
    "DS_Store",
    "rdb",
    "pack",
    "idx",
    "glb",
    "gltf",
    "len",
    "otf",
    "unitypackage",
    "ttf",
    "xz",
    "pcm",
    "opus",
]
ANTI_FOMATS = tuple(IMAGE + VIDEO + DOC + AUDIO + ARCHIVE + OTHERS)


def segment_blocks(content):
    """
    Segments the cells of a Jupyter notebook into their source and output components.

    Args:
        content (dict): The loaded JSON content of a Jupyter notebook.

    Returns:
        tuple: A tuple containing a list of [source, output] pairs and a list of cell types.
    """
    cells = []
    cell_types = []
    for cell in content["cells"]:
        if len(cell["source"]) > 0:
            output = "_____no_output_____"
            if "outputs" in cell.keys():
                if len(cell["outputs"]) > 0:
                    if "text" in cell["outputs"][0].keys():
                        output = cell["outputs"][0]["text"]
            cells.append(["".join(cell["source"]), "".join(output)])
            cell_types.append(cell["cell_type"])
    return cells, cell_types


def segment(sample):
    """
    Processes a Jupyter notebook sample, segmenting it into groups of cells and types.

    Args:
        sample (str): The JSON string of a Jupyter notebook.

    Returns:
        str: The parsed and formatted content extracted from the notebook.
    """
    try:
        content = json.loads(sample)
        if "py" in json.dumps(content["metadata"]):
            cells, types = segment_blocks(content)

            cell_type_groups = [list(g) for k, g in itertools.groupby(types)]
            cell_types = [k for k, g in itertools.groupby(types)]
            cell_groups = []

            group_start = 0
            for g in cell_type_groups:
                cell_groups.append(cells[group_start : group_start + len(g)])
                group_start += len(g)
        else:
            cell_groups = [[["empty"]]]
            cell_types = ["empty"]
            cell_type_groups = [["empty"]]

    except:  # noqa: E722
        cell_groups = [[["empty"]]]
        cell_types = ["empty"]
        cell_type_groups = [["empty"]]

    content = parse_data(cell_groups, cell_types)

    return content


def clean_markdown(text):
    """
    Cleans markdown text by removing HTML tags, newlines, and hash symbols.

    Args:
        text (str): The markdown text to clean.

    Returns:
        str: The cleaned markdown text.
    """
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\n+", "", text)
    text = text.replace("#", "")
    return text


def parse_data(cells, types):
    """Parse data into markdown-code pairs

    Args:
        cells (list): List of cell groups.
        types (list): List of cell types.

    Returns:
        str: The formatted content string with markdown, code, and output.
    """

    content = ""
    if len(types) > 0:
        if types[0] == "code":
            # add dummy markdown
            cells.insert(0, [["empty"]])
            types.insert(0, "markdown")
        if len(types) > 0:
            if types[-1] == "markdown":
                cells = cells[:-1]
                types = types[:-1]

            if len(cells) % 2 == 0:
                inner_markdowns = [cells[j] for j in range(len(cells)) if j % 2 == 0]
                inner_code_snippets = [
                    cells[j + 1] for j in range(len(cells) - 1) if j % 2 == 0
                ]

                content += "<jupyter_start>"
                for markdown_block, code_snippet in zip(
                    inner_markdowns, inner_code_snippets
                ):
                    markdown_block = " ".join(
                        [clean_markdown(block[0]) for block in markdown_block]
                    )
                    code = "\n".join([snippet[0] for snippet in code_snippet])
                    output = [snippet[1] for snippet in code_snippet][-1]
                    content += build_content(markdown_block, code, output)
    return content


def build_content(markdown, code, output):
    """
    Builds a formatted string for a markdown, code, and output block.

    Args:
        markdown (str): The markdown text.
        code (str): The code snippet.
        output (str): The output of the code cell.

    Returns:
        str: The formatted content string.
    """
    if len(output) > 1000:
        output_str = output[:1000] + "[...]"
    elif output == "_____no_output_____":
        output_str = "<empty_output>"
    else:
        output_str = output
    if markdown.strip() != "empty":
        content = f"<jupyter_text>{markdown.strip()}<jupyter_code>{code.strip()}<jupyter_output>{output_str.strip()}"
    else:
        content = f"<jupyter_code>{code.strip()}<jupyter_output>{output_str.strip()}"
    return content


class PersonalCopilotDatasetReader(BaseDiskReader):
    """
    Custom dataset reader for the PersonalCopilot project.

    This reader filters out unwanted file types, processes Jupyter notebooks with special
    segmentation, and yields documents with extracted text and metadata for downstream
    data processing pipelines.
    """
    name = "👾 PersonalCopilot"

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: str = None,
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        """
        Initializes the PersonalCopilotDatasetReader.

        Args:
            data_folder (DataFolderLike): The folder containing the data to read.
            paths_file (str, optional): File containing specific paths to read.
            limit (int, optional): Maximum number of files to read.
            skip (int, optional): Number of files to skip.
            file_progress (bool, optional): Whether to show file progress.
            doc_progress (bool, optional): Whether to show document progress.
            adapter (Callable, optional): Adapter function for processing files.
            text_key (str, optional): Key for text in the output dictionary.
            id_key (str, optional): Key for id in the output dictionary.
            default_metadata (dict, optional): Default metadata to attach to documents.
            recursive (bool, optional): Whether to read files recursively.
            glob_pattern (str, optional): Glob pattern for file selection.
            shuffle_files (bool, optional): Whether to shuffle files before reading.
        """
        super().__init__(
            data_folder,
            paths_file,
            limit,
            skip,
            file_progress,
            doc_progress,
            adapter,
            text_key,
            id_key,
            default_metadata,
            recursive,
            glob_pattern,
            shuffle_files,
        )
        self.empty_warning = False

    def read_file(self, filepath: str):
        """
        Reads a file, filters out unwanted formats, processes Jupyter notebooks,
        and yields a document with extracted text and metadata.

        Args:
            filepath (str): The path to the file to read.

        Yields:
            Document: A datatrove Document object with text and metadata.
        """
        try:
            if filepath.endswith(ANTI_FOMATS) or any(
                k in filepath for k in [".git", "__pycache__", "xcodeproj"]
            ):
                content = ""
            else:
                with self.data_folder.open(filepath, "r", encoding="utf-8") as file:
                    content = file.read()
                    if filepath.endswith("ipynb"):
                        content = segment(content)
        except Exception:
            content = ""

        if not content:
            content = "remove"
        data = {"text": content}
        with self.track_time():
            document = self.get_document_from_dict(data, filepath, 0)
            document.metadata["file_path"] = document.metadata["file_path"].split(
                self.data_folder.path
            )[-1][1:]
            document.metadata["repo_id"] = document.metadata["file_path"].split("/")[0]
        yield document
