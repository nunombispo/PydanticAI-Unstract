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
import psycopg
from datetime import datetime, UTC


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


def process_dataframe(data: Dict[str, Any], action: str = 'sum', split_year: int = 2025) -> str:
    """Process any nested dictionary data using pandas and return a processed DataFrame with aggregated values.
    
    Args:
        data (Dict[str, Any]): The input data dictionary
        action (str): The aggregation action to perform ('sum' or 'avg')
        split_year (int): The year to split the data before and after (default: 2025)
    
    Returns:
        JSON string containing the aggregated data
    """
    print(f"process_dataframe called with action={action}, split_year={split_year}")
    
    # Validate action parameter
    if action not in ['sum', 'avg']:
        raise ValueError("Action must be either 'sum' or 'avg'")
    
    # Parse JSON if data is a string
    if isinstance(data, str):
        try:
            raw_data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
    else:
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
    
    # Return a JSON of the DataFrame
    return result_df.to_json(orient='records', indent=2)


def _save_data_to_database(data: Dict[str, Any], action: str = 'sum', split_year: int = 2025) -> Dict[str, Any]:
    """Save the processed data to the database using direct psycopg2 connection.
    
    Args:
        data (Dict[str, Any]): The processed data to save
        action (str): The aggregation action ('sum' or 'avg')
        split_year (int): The year used for splitting the data
        
    Returns:
        Dict[str, Any]: Result of the database operation including status and inserted records
    """
    print("save_data_to_database called")
    try:
        # Parse JSON if data is a string
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}")

        # Get the connection string from the environment variable
        connection_string = os.getenv('NEON_DATABASE_URL')
        
        with psycopg.connect(connection_string) as conn:
            # Open a cursor to perform database operations
            with conn.cursor() as cur:
                # Create tables if they don't exist
                if action == 'sum':
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS economic_report_sum (
                            id SERIAL PRIMARY KEY,
                            category VARCHAR NOT NULL,
                            metric VARCHAR NOT NULL,
                            sum_before FLOAT NOT NULL,
                            sum_after FLOAT NOT NULL,
                            split_year INTEGER NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                else:  # action == 'avg'
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS economic_report_avg (
                            id SERIAL PRIMARY KEY,
                            category VARCHAR NOT NULL,
                            metric VARCHAR NOT NULL,
                            avg_before FLOAT NOT NULL,
                            avg_after FLOAT NOT NULL,
                            split_year INTEGER NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
        
                # Prepare data for insertion
                records_inserted = 0
                for record in data['records']:
                    if action == 'sum':
                        category = record['Category']
                        metric = record['Metric']
                        sum_before = record[f'{action.title()}_Before_{split_year}']
                        sum_after = record[f'{action.title()}_After_{split_year}']
                        created_at = datetime.now(UTC)
                        
                        insert_query = """
                            INSERT INTO economic_report_sum 
                            (category, metric, sum_before, sum_after, split_year, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """
                        cur.execute(insert_query, (category, metric, sum_before, sum_after, split_year, created_at))
                    else:  # action == 'avg'
                        category = record['Category']
                        metric = record['Metric']
                        avg_before = record[f'{action.title()}_Before_{split_year}']
                        avg_after = record[f'{action.title()}_After_{split_year}']
                        created_at = datetime.now(UTC)
                        
                        insert_query = """
                            INSERT INTO economic_report_avg 
                            (category, metric, avg_before, avg_after, split_year, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """
                        cur.execute(insert_query, (category, metric, avg_before, avg_after, split_year, created_at))
                    
                    # Increment the number of records inserted
                    records_inserted += 1
                    
                # Commit the transaction
                conn.commit()
                
                # Prepare result
                result = {
                    'status': 'success',
                    'action': action,
                    'split_year': split_year,
                    'records_inserted': records_inserted,
                    'timestamp': datetime.now(UTC).isoformat()
                }
                
                return result
        
    except Exception as e:
        # Rollback in case of error
        if 'conn' in locals() and conn is not None:
            conn.rollback()
        return {
            'status': 'error',
            'error': str(e),
            'action': action,
            'split_year': split_year,
            'timestamp': datetime.now(UTC).isoformat()
        }
    finally:
        # Close the cursor and connection if they exist
        if 'cur' in locals() and cur is not None:
            cur.close()
        if 'conn' in locals() and conn is not None:
            conn.close()

# Define the function to save the data to the database for action 'sum'
def save_data_to_database_sum(data: Dict[str, Any], split_year: int = 2025) -> None:
    """Save the data to the database for action 'sum'."""
    _save_data_to_database(data, 'sum', split_year)


# Define the function to save the data to the database for action 'avg'
def save_data_to_database_avg(data: Dict[str, Any], split_year: int = 2025) -> None:
    """Save the data to the database for action 'avg'."""
    _save_data_to_database(data, 'avg', split_year)


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
                You can also use 'save_data_to_database_sum' to save the data to the database for action 'sum'.
                You can also use 'save_data_to_database_avg' to save the data to the database for action 'avg'.
            """
        )

    # Define the messages to send to the agent
    messages = [
            "You have tools available if you need to extract the text from the PDF file.",
            "You have tools available if you need to process data using Pandas.",
            "You have tools available if you need to save the data to the database for action 'sum'.",
            "You have tools available if you need to save the data to the database for action 'avg'.",
            "Extract the text from the PDF file and process the data using Pandas to return the average of year 2025 and save the result to the database (make sure to pass a list with records).",
            "Return the result in a JSON format."
        ]

    # Define the dependencies to send to the agent
    deps = BinaryContent(data=pdf_path.read_bytes(), media_type='application/pdf')

    # Define the agent
    agent = Agent(
        os.getenv('MISTRAL_MODEL'),  
        api_key=os.getenv('MISTRAL_API_KEY'),
        tools=[extract_pdf_text, process_dataframe, save_data_to_database_sum, save_data_to_database_avg], 
        system_prompt=system_prompt
    )

    # Run the async function
    asyncio.run(run_agent(messages, deps, agent))