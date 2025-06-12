# PDF Text Extraction Tool with Ollama

This project implements a PydanticAI agent tool that extracts text from PDF files and processes it using Ollama's Llama model.

## Prerequisites

1. Python 3.8 or higher
2. Ollama installed and running locally (https://ollama.ai/)
3. Llama2 model pulled in Ollama (`ollama pull llama2`)

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The tool can be used in two ways:

1. As a standalone script:

```python
import asyncio
from pdf_extractor_agent import process_pdf

async def main():
    result = await process_pdf("path/to/your.pdf")
    print("Extracted text:", result.extracted_text)
    print("Processed text:", result.processed_text)

asyncio.run(main())
```

2. As part of a PydanticAI agent:

```python
from pdf_extractor_agent import PDFExtractionTool, Agent

agent = Agent(
    name="PDF Processing Agent",
    tools=[PDFExtractionTool()]
)

# Use the agent with your PDF data
result = await agent.run_tool(
    "pdf_text_extractor",
    {
        "pdf_binary": pdf_binary_data,
        "ollama_model": "llama2",  # Optional, defaults to "llama2"
        "ollama_base_url": "http://localhost:11434"  # Optional, defaults to localhost
    }
)
```

## Configuration

The tool accepts the following parameters:

- `pdf_binary`: The PDF file as binary data (required)
- `ollama_model`: The Ollama model to use (default: "llama2")
- `ollama_base_url`: Base URL for Ollama API (default: "http://localhost:11434")

## Output

The tool returns an object with two fields:

- `extracted_text`: The raw text extracted from the PDF
- `processed_text`: The text processed by the Ollama model (may be None if processing fails)

## Error Handling

The tool includes basic error handling:

- PDF extraction errors will raise exceptions
- Ollama processing errors will be caught and logged, with `processed_text` set to None
