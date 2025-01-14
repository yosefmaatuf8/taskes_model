from odf.opendocument import OpenDocumentText, load
from odf.text import P
import os
import tiktoken

# Initialize tokenizer for GPT-4
tokenizer = tiktoken.encoding_for_model("gpt-4")

def save_to_odt(text, filename):
    try:
        # Create an ODT document
        doc = OpenDocumentText()

        # Split the transcription into individual speaker lines
        lines = text.split("\n")

        # Add each line as a paragraph, with a line break between speakers
        for line in lines:
            # Add a line break after each speaker's dialogue
            if line.strip():  # Avoid adding empty paragraphs
                doc.text.addElement(P(text=line))
                doc.text.addElement(P(text=""))  # Add a blank line (line break)

        # Save the document
        doc.save(filename)

        print(f"File saved successfully as: {os.path.abspath(filename)}")

    except Exception as e:
        print(f"Error: {str(e)}")

def extract_text_from_odt(filename):
    try:
        # Load the ODT file
        doc = load(filename)

        # Extract all text paragraphs
        paragraphs = []
        for element in doc.getElementsByType(P):
            paragraphs.append(str(element))

        # Join paragraphs with newline
        return "\n".join(paragraphs)

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def split_text(text, max_tokens):
    """Split text into chunks within the token limit."""
    words = text.split()  # Split by words
    chunks = []
    current_chunk = []
    overlap = 15  # Number of overlapping words

    for word in words:
        # Check if adding the word exceeds token limit
        if len(tokenizer.encode(" ".join(current_chunk + [word]))) > max_tokens:
            chunks.append(" ".join(current_chunk))  # Save current chunk
            # Start new chunk with the last words of the previous chunk
            current_chunk = current_chunk[-overlap:] + [word]
        else:
            current_chunk.append(word)

    if current_chunk:  # Add remaining words
        chunks.append(" ".join(current_chunk))

    return chunks