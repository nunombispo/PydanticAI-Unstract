import os
import random
from io import BytesIO
from typing import BinaryIO
from dotenv import load_dotenv
from pydantic_ai import Agent, BinaryContent, RunContext, ModelRetry
from pypdf import PdfReader
from pathlib import Path

load_dotenv()

def extract_pdf_text(ctx: RunContext[str]) -> str:
    """Extract text from PDF binary data."""
    print("Tool called")
    bytes = ctx.deps.data
    pdf_binary = ctx.deps.data  # Get binary data from context
    pdf_file = BytesIO(pdf_binary)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

pdf_path = Path('P&G annual report.pdf')

agent = Agent(
    'mistral:mistral-large-latest',  
    api_key=os.getenv('GEMINI_API_KEY'),
    tools=[extract_pdf_text], 
    system_prompt=(
        """
            You are a helpful assistant that has access to a PDF file.
            Make sure to use the tool 'extract_pdf_text' to extract the text from the PDF file.
        """
    ),
)

# Example usage
if __name__ == "__main__":
    
    async def run_agent():
        nodes = []
        # Begin an AgentRun, which is an async-iterable over the nodes of the agent's graph
        async with agent.iter(
            [
                "You have tools available if you need to extract the text from the PDF file."
                "What is the contents of the pdf file about?",
            ],
            deps=BinaryContent(data=pdf_path.read_bytes(), media_type='application/pdf')
        ) as agent_run:
            async for node in agent_run:
                # Each node represents a step in the agent's execution
                print(node)
                print("\n\n")
        print(agent_run.result.output)

    # Run the async function
    import asyncio
    asyncio.run(run_agent())