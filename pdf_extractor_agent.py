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
    
    return """
{
  "national_accounts": [
    {
      "Real GDP (production)": [
        {
          "2020": -1.4
        },
        {
          "2021": 5.6
        },
        {
          "2022": 2.4
        },
        {
          "2023": 0.6
        },
        {
          "2024": 1
        },
        {
          "2025": 2
        },
        {
          "2026": 2.4
        },
        {
          "2027": 2.4
        },
        {
          "2028": 2.4
        },
        {
          "2029": 2.4
        }
      ]
    },
    {
      "Domestic demand": [
        {
          "2020": -1.7
        },
        {
          "2021": 10.1
        },
        {
          "2022": 3.4
        },
        {
          "2023": -1.5
        },
        {
          "2024": -0.4
        },
        {
          "2025": 1.5
        },
        {
          "2026": 2
        },
        {
          "2027": 2.1
        },
        {
          "2028": 2.2
        },
        {
          "2029": 2.1
        }
      ]
    },
    {
      "Private consumption": [
        {
          "2020": -1.7
        },
        {
          "2021": 7.7
        },
        {
          "2022": 3.2
        },
        {
          "2023": 0.3
        },
        {
          "2024": -1.6
        },
        {
          "2025": 2
        },
        {
          "2026": 2.1
        },
        {
          "2027": 2.3
        },
        {
          "2028": 2.4
        },
        {
          "2029": 2.3
        }
      ]
    },
    {
      "Public consumption": [
        {
          "2020": 6.7
        },
        {
          "2021": 7.8
        },
        {
          "2022": 4.9
        },
        {
          "2023": -1.1
        },
        {
          "2024": -1.1
        },
        {
          "2025": 0
        },
        {
          "2026": 0.6
        },
        {
          "2027": 0.4
        },
        {
          "2028": 0.4
        },
        {
          "2029": 0.4
        }
      ]
    },
    {
      "Investment": [
        {
          "2020": -7.8
        },
        {
          "2021": 18.1
        },
        {
          "2022": 2
        },
        {
          "2023": -5.1
        },
        {
          "2024": 0.3
        },
        {
          "2025": 1.5
        },
        {
          "2026": 3
        },
        {
          "2027": 3
        },
        {
          "2028": 3
        },
        {
          "2029": 2.9
        }
      ]
    },
    {
      "Public": [
        {
          "2020": 4
        },
        {
          "2021": 7.9
        },
        {
          "2022": -6.4
        },
        {
          "2023": 4.9
        },
        {
          "2024": 2.5
        },
        {
          "2025": 1.3
        },
        {
          "2026": 2.3
        },
        {
          "2027": 2.5
        },
        {
          "2028": 2.8
        },
        {
          "2029": 2.8
        }
      ]
    },
    {
      "Private": [
        {
          "2020": -7.4
        },
        {
          "2021": 13.5
        },
        {
          "2022": 6.3
        },
        {
          "2023": -2.6
        },
        {
          "2024": -4.1
        },
        {
          "2025": 1.5
        },
        {
          "2026": 3.2
        },
        {
          "2027": 3.1
        },
        {
          "2028": 3.1
        },
        {
          "2029": 2.9
        }
      ]
    },
    {
      "Private business": [
        {
          "2020": -9.3
        },
        {
          "2021": 15.7
        },
        {
          "2022": 9.6
        },
        {
          "2023": -1.9
        },
        {
          "2024": -4.5
        },
        {
          "2025": 1.4
        },
        {
          "2026": 3.4
        },
        {
          "2027": 3.4
        },
        {
          "2028": 3.4
        },
        {
          "2029": 3.1
        }
      ]
    },
    {
      "Dwelling": [
        {
          "2020": -3.1
        },
        {
          "2021": 9
        },
        {
          "2022": -0.9
        },
        {
          "2023": -4.2
        },
        {
          "2024": -3
        },
        {
          "2025": 1.9
        },
        {
          "2026": 2.8
        },
        {
          "2027": 2.4
        },
        {
          "2028": 2.4
        },
        {
          "2029": 2.4
        }
      ]
    },
    {
      "Inventories (contribution to growth, percent)": [
        {
          "2020": -0.8
        },
        {
          "2021": 1.4
        },
        {
          "2022": -0.4
        },
        {
          "2023": -1.1
        },
        {
          "2024": 0.7
        },
        {
          "2025": 0
        },
        {
          "2026": 0
        },
        {
          "2027": 0
        },
        {
          "2028": 0
        },
        {
          "2029": 0
        }
      ]
    },
    {
      "Net exports (contribution to growth, percent)": [
        {
          "2020": 1.5
        },
        {
          "2021": -4.8
        },
        {
          "2022": -1.5
        },
        {
          "2023": 2.2
        },
        {
          "2024": 1.5
        },
        {
          "2025": 0.3
        },
        {
          "2026": 0.3
        },
        {
          "2027": 0.1
        },
        {
          "2028": 0.1
        },
        {
          "2029": 0.1
        }
      ]
    },
    {
      "Real gross domestic income": [
        {
          "2020": -0.7
        },
        {
          "2021": 5.1
        },
        {
          "2022": 1.3
        },
        {
          "2023": 0
        },
        {
          "2024": 1.4
        },
        {
          "2025": 2.1
        },
        {
          "2026": 2.5
        },
        {
          "2027": 2.5
        },
        {
          "2028": 2.5
        },
        {
          "2029": 2.5
        }
      ]
    },
    {
      "Investment (percent of GDP)": [
        {
          "2020": 22.1
        },
        {
          "2021": 25
        },
        {
          "2022": 26
        },
        {
          "2023": 24.4
        },
        {
          "2024": 24.3
        },
        {
          "2025": 24.2
        },
        {
          "2026": 24.4
        },
        {
          "2027": 24.4
        },
        {
          "2028": 24.4
        },
        {
          "2029": 24.5
        }
      ]
    },
    {
      "Public": [
        {
          "2020": 5.5
        },
        {
          "2021": 5.7
        },
        {
          "2022": 5.4
        },
        {
          "2023": 5.7
        },
        {
          "2024": 5.7
        },
        {
          "2025": 5.7
        },
        {
          "2026": 5.7
        },
        {
          "2027": 5.6
        },
        {
          "2028": 5.6
        },
        {
          "2029": 5.6
        }
      ]
    },
    {
      "Private": [
        {
          "2020": 16.6
        },
        {
          "2021": 19.4
        },
        {
          "2022": 20.6
        },
        {
          "2023": 18.7
        },
        {
          "2024": 18.6
        },
        {
          "2025": 18.6
        },
        {
          "2026": 18.7
        },
        {
          "2027": 18.7
        },
        {
          "2028": 18.8
        },
        {
          "2029": 18.9
        }
      ]
    },
    {
      "Savings (gross, percent of GDP)": [
        {
          "2020": 21.1
        },
        {
          "2021": 19.2
        },
        {
          "2022": 17.2
        },
        {
          "2023": 17.5
        },
        {
          "2024": 18.3
        },
        {
          "2025": 18.9
        },
        {
          "2026": 19.6
        },
        {
          "2027": 20
        },
        {
          "2028": 20.3
        },
        {
          "2029": 20.8
        }
      ]
    },
    {
      "Public": [
        {
          "2020": -4.3
        },
        {
          "2021": -3.2
        },
        {
          "2022": -3.5
        },
        {
          "2023": -3.5
        },
        {
          "2024": -3.5
        },
        {
          "2025": -2.6
        },
        {
          "2026": -1.7
        },
        {
          "2027": -1.1
        },
        {
          "2028": -0.4
        },
        {
          "2029": -0.1
        }
      ]
    },
    {
      "Private": [
        {
          "2020": 25.5
        },
        {
          "2021": 22.4
        },
        {
          "2022": 20.7
        },
        {
          "2023": 21
        },
        {
          "2024": 21.8
        },
        {
          "2025": 21.4
        },
        {
          "2026": 21.3
        },
        {
          "2027": 21
        },
        {
          "2028": 20.7
        },
        {
          "2029": 20.9
        }
      ]
    },
    {
      "Potential output": [
        {
          "2020": 1.6
        },
        {
          "2021": 1.5
        },
        {
          "2022": 1.9
        },
        {
          "2023": 2.1
        },
        {
          "2024": 2.3
        },
        {
          "2025": 2.3
        },
        {
          "2026": 2.2
        },
        {
          "2027": 2.2
        },
        {
          "2028": 2.2
        },
        {
          "2029": 2.2
        }
      ]
    },
    {
      "Output gap (percent of potential)": [
        {
          "2020": -2.3
        },
        {
          "2021": 1.7
        },
        {
          "2022": 2.1
        },
        {
          "2023": 0.6
        },
        {
          "2024": -0.5
        },
        {
          "2025": -0.9
        },
        {
          "2026": -0.7
        },
        {
          "2027": -0.5
        },
        {
          "2028": -0.2
        },
        {
          "2029": 0
        }
      ]
    }
  ],
  "labor_market": [
    {
      "Employment": [
        {
          "2020": 1.3
        },
        {
          "2021": 2.2
        },
        {
          "2022": 1.7
        },
        {
          "2023": 3.1
        },
        {
          "2024": 1.1
        },
        {
          "2025": 1.1
        },
        {
          "2026": 1.6
        },
        {
          "2027": 1.7
        },
        {
          "2028": 1.7
        },
        {
          "2029": 1.6
        }
      ]
    },
    {
      "Unemployment (percent of labor force, ann. average)": [
        {
          "2020": 4.6
        },
        {
          "2021": 3.8
        },
        {
          "2022": 3.3
        },
        {
          "2023": 3.7
        },
        {
          "2024": 5
        },
        {
          "2025": 5.4
        },
        {
          "2026": 5.2
        },
        {
          "2027": 5
        },
        {
          "2028": 4.7
        },
        {
          "2029": 4.5
        }
      ]
    },
    {
      "Wages (nominal percent change)": [
        {
          "2020": 3.8
        },
        {
          "2021": 3.8
        },
        {
          "2022": 6.5
        },
        {
          "2023": 7
        },
        {
          "2024": 4.8
        },
        {
          "2025": 3.9
        },
        {
          "2026": 3.7
        },
        {
          "2027": 3.2
        },
        {
          "2028": 3
        },
        {
          "2029": 3
        }
      ]
    }
  ]
}
        """
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


def process_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """Process any nested dictionary data using pandas and return a processed DataFrame."""
    print("Data processing tool called")
    
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
    
    # Sort by Category, Metric, and Year
    df = df.sort_values(['Category', 'Metric', 'Year'])
    
    # Create a pivot table with proper handling of the index
    try:
        df_pivot = pd.pivot_table(
            data=df,
            values='Value',
            index=['Category', 'Metric'],
            columns='Year',
            aggfunc='first'  # Use first value if there are duplicates
        )
        
        # Reset index to make Category and Metric regular columns
        df_pivot = df_pivot.reset_index()
        
        # Fill NaN values with 'N/A'
        df_pivot = df_pivot.fillna('N/A')
        
        print("\nProcessed DataFrame Preview:")
        print(df_pivot.head())
        
        # Return a JSON of the DataFrame
        return df_pivot.to_json()
    except Exception as e:
        print(f"Error creating pivot table: {str(e)}")
        # Return the original DataFrame if pivot fails
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