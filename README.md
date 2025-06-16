# PDF Text Extraction and Data Processing Tool

This project implements a PydanticAI agent tool that extracts text from PDF files using Unstract API, processes data using Pandas, and integrates with a PostgreSQL database for data storage and analysis.

The agent is powered by Mistral's AI model.

## Features

- PDF text extraction using Unstract API
- Data processing and analysis using Pandas
- Integration with Mistral AI model for agent operations
- PydantiAI for AI agent processing
- PostgreSQL database integration for data persistence
- Support for data aggregation (sum/avg) with year-based splitting
- Asynchronous processing capabilities

## Prerequisites

1. Python 3.8 or higher
2. Unstract API access (for PDF extraction)
3. Mistral API access (for agent operations)
4. PostgreSQL database (or Neon database)

## Environment Setup

Create a `.env` file in the project root with the following variables:

```env
UNSTRACT_API_URL=your_unstract_api_url
UNSTRACT_API_KEY=your_unstract_api_key
MISTRAL_MODEL=your_mistral_model
MISTRAL_API_KEY=your_mistral_api_key
NEON_DATABASE_URL=your_database_connection_string
```

## Installation

1. Clone this repository
2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Adjust the `messages` for the desired processing by the AI Agent.

Then execute the script with:

```
python pdf_extractor_agent.py
```

## Database Schema

The tool creates two tables in the database:

1. `economic_report_sum`:

   - id (SERIAL PRIMARY KEY)
   - category (VARCHAR)
   - metric (VARCHAR)
   - sum_before (FLOAT)
   - sum_after (FLOAT)
   - split_year (INTEGER)
   - created_at (TIMESTAMP)

2. `economic_report_avg`:
   - id (SERIAL PRIMARY KEY)
   - category (VARCHAR)
   - metric (VARCHAR)
   - avg_before (FLOAT)
   - avg_after (FLOAT)
   - split_year (INTEGER)
   - created_at (TIMESTAMP)

## Error Handling

The tool includes comprehensive error handling:

- PDF extraction errors are caught and logged
- Database operations include transaction management with rollback on failure
- Data processing errors include validation and type checking
- All operations include proper resource cleanup
- Detailed error messages are provided for debugging

## License

This project is licensed under the MIT License.
