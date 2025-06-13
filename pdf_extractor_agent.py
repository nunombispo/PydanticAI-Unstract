import os
from typing import Dict, Any
import pandas as pd
from dotenv import load_dotenv
from pydantic_ai import Agent, BinaryContent, RunContext
from pathlib import Path
import requests
import asyncio
import numpy as np


# Load the environment variables
load_dotenv()

# Define the function to extract text from the PDF file
def extract_pdf_text(ctx: RunContext[str]) -> str:
    """Extract text from PDF binary data."""
    print("extract_pdf_text called")
    
    try:
        # Get the binary data from the context
        pdf_binary = ctx.deps.data
        # Write the pdf file to a temporary file
        filepath = 'temp.pdf'
        with open(filepath, 'wb') as f:
            f.write(pdf_binary)

        # Define the API URL and headers
        api_url = os.getenv('UNSTRACT_API_URL')
        headers = {
            'Authorization': 'Bearer ' + os.getenv('UNSTRACT_API_KEY')
        }
        # Define the payload
        payload = {'timeout': 300, 'include_metadata': False}
        # Define the files
        files=[('files',('file',open(filepath,'rb'),'application/octet-stream'))]
        # Make the request
        response = requests.post(api_url, headers=headers, data=payload, files=files)
        # Return the response
        return response.json()['message']['result'][0]['result']['output']
    except Exception as e:
        print(e)
        return "Error extracting text from PDF"


def flatten_nested_dict(nested_dict, parent_key='', sep='_'):
    """Flatten a nested dictionary into a list of tuples (key_path, value)."""
    items = []
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_nested_dict(value, new_key, sep=sep).items())
        elif isinstance(value, list):
            # Handle lists of dictionaries
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.extend(flatten_nested_dict(item, f"{new_key}_{i}", sep=sep).items())
                else:
                    items.append((f"{new_key}_{i}", item))
        else:
            items.append((new_key, value))
    return dict(items)


def process_dataframe(ctx: RunContext[Dict[str, Any]]) -> pd.DataFrame:
    """Process any nested dictionary data using pandas and return a processed DataFrame."""
    print("Data processing tool called")
    data = ctx.deps.data  # Get data from context
    
    try:
        # First attempt: try direct DataFrame creation
        df = pd.DataFrame(data)
    except ValueError:
        try:
            # Second attempt: flatten nested structure and create DataFrame
            flattened_data = flatten_nested_dict(data)
            df = pd.DataFrame([flattened_data])
            
            # If we have a single row, try to reshape it
            if len(df) == 1:
                # Melt the DataFrame to get a long format
                df = df.melt(var_name='Metric', value_name='Value')
                
                # Try to extract year/period from metric names if they exist
                df['Period'] = df['Metric'].str.extract(r'(\d{4})').fillna('Unknown')
                df['Metric'] = df['Metric'].str.replace(r'_\d{4}', '', regex=True)
                
                # Pivot to get periods as columns if we have period information
                if not df['Period'].eq('Unknown').all():
                    df = df.pivot(index='Metric', columns='Period', values='Value').reset_index()
        
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            # Last resort: create a simple DataFrame with the raw data
            df = pd.DataFrame({'raw_data': [str(data)]})
    
    # Basic data cleaning
    df = df.fillna('N/A')  # Replace NaN values
    df = df.replace([np.inf, -np.inf], 'N/A')  # Replace infinite values
    
    print("\nProcessed DataFrame Preview:")
    print(df.head())
    print("\nDataFrame Info:")
    print(df.info())
    
    return df


# Define the async function to run the agent
async def run_agent(messages, deps, agent):
    # Begin an AgentRun, which is an async-iterable over the nodes of the agent's graph
    print("-" * 100)
    print("Running agent")
    async with agent.iter(messages, deps=deps) as agent_run:
        async for node in agent_run:
            # Each node represents a step in the agent's execution
            print(node)
            print("\n")
    print("-" * 100)
    print("Agent run completed")
    # Print the result
    print("-" * 100)
    print(agent_run.result.output)


# Usage example
if __name__ == "__main__":
    # Get the path to the PDF file
    pdf_path = Path('new-zealand-economic-report-1.pdf')

    # Define the system prompt
    system_prompt=(
            """
                You are a helpful assistant that has access to a PDF file and can process data using Pandas.
                Make sure to use the tool 'extract_pdf_text' to extract the text from the PDF file.
                You can also use 'process_dataframe' to process data using Pandas.
            """
        )

    # Define the messages to send to the agent
    messages = [
            "You have tools available if you need to extract the text from the PDF file.",
            "You have tools available if you need to process data using Pandas.",
            "Extract the text from the PDF file and process the data using Pandas.",
        ]

    # Define the dependencies to send to the agent
    deps = BinaryContent(data=pdf_path.read_bytes(), media_type='application/pdf')

    # Define the agent
    agent = Agent(
        os.getenv('MISTRAL_MODEL'),  
        api_key=os.getenv('MISTRAL_API_KEY'),
        tools=[extract_pdf_text, process_dataframe], 
        system_prompt=system_prompt
    )

    # Run the async function
    asyncio.run(run_agent(messages, deps, agent))