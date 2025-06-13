import json
import os
import time
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


def process_dataframe(data: Dict[str, Any], action: str = 'sum', split_year: int = 2025) -> pd.DataFrame:
    """Process any nested dictionary data using pandas and return a processed DataFrame with aggregated values.
    
    Args:
        data (Dict[str, Any]): The input data dictionary
        action (str): The aggregation action to perform ('sum' or 'avg')
        split_year (int): The year to split the data before and after (default: 2025)
    
    Returns:
        str: JSON string containing the aggregated data
    """
    print(f"Data processing tool called with action={action}, split_year={split_year}")
    
    # Validate action parameter
    if action not in ['sum', 'avg']:
        raise ValueError("Action must be either 'sum' or 'avg'")
    
    # Get the data from the context
    raw_data = data
    
    # Initialize lists to store the data
    metrics = []
    categories = []
    years = []
    values = []
    
    # Process each category (labor_market, national_accounts)
    for category, metrics_list in raw_data.items():
        # Process each metric dictionary in the list
        for metric_dict in metrics_list:
            # Each metric_dict has one key-value pair
            for metric_name, year_list in metric_dict.items():
                # Process each year dictionary in the list
                for year_dict in year_list:
                    # Each year_dict has one key-value pair
                    for year, value in year_dict.items():
                        metrics.append(metric_name)
                        categories.append(category)
                        years.append(int(year))  # Convert year to integer
                        values.append(float(value))  # Convert value to float
    
    # Create DataFrame
    df = pd.DataFrame({
        'Category': categories,
        'Metric': metrics,
        'Year': years,
        'Value': values
    })
    
    # Create separate DataFrames for before and after split_year
    df_before = df[df['Year'] < split_year].groupby(['Category', 'Metric'])['Value']
    df_after = df[df['Year'] >= split_year].groupby(['Category', 'Metric'])['Value']
    
    # Apply the specified action
    if action == 'sum':
        df_before = df_before.sum().reset_index()
        df_after = df_after.sum().reset_index()
        before_col = f'Sum_Before_{split_year}'
        after_col = f'Sum_After_{split_year}'
    else:  # action == 'avg'
        df_before = df_before.mean().reset_index()
        df_after = df_after.mean().reset_index()
        before_col = f'Avg_Before_{split_year}'
        after_col = f'Avg_After_{split_year}'
    
    # Rename the Value columns
    df_before = df_before.rename(columns={'Value': before_col})
    df_after = df_after.rename(columns={'Value': after_col})
    
    # Merge the two DataFrames
    result_df = pd.merge(df_before, df_after, on=['Category', 'Metric'])
    
    # Sort by Category and Metric
    result_df = result_df.sort_values(['Category', 'Metric'])
    
    # Round numeric columns to 2 decimal places
    numeric_cols = result_df.select_dtypes(include=['float64']).columns
    result_df[numeric_cols] = result_df[numeric_cols].round(2)
    
    print(f"\nSummary of {action.title()} Values Before and After {split_year}:")
    print(result_df)
    
    # Return a JSON of the DataFrame
    return result_df.to_json(orient='records', indent=2)


# Define the async function to run the agent
async def run_agent(messages, deps, agent):
    # Begin an AgentRun, which is an async-iterable over the nodes of the agent's graph
    print("-" * 100)
    start_time = time.time()
    print("Running agent")
    print("-" * 100)
    async with agent.iter(messages, deps=deps) as agent_run:
        async for node in agent_run:
            # Each node represents a step in the agent's execution
            print(node)
            print("\n")
    end_time = time.time()
    print("-" * 100)
    print(f"Agent run completed in {end_time - start_time} seconds")
    print("-" * 100)
    # Print the result
    print("\n")
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
            "Extract the text from the PDF file and process the data using Pandas to return the sum of year 2023.",
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