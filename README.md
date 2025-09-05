py -m uvicorn app.main:app --reload

# Discord SUI Analyzer

This project is a FastAPI application that analyzes Discord chat logs to identify technical discussions and extract solutions.

## Getting Started

### Installation

1.  **Clone the repository and navigate into the project directory.**

2.  **Create and activate a virtual environment:**
    ```shell
    # On Windows
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```shell
    pip install -r requirements.txt
    ```

4.  **Set up your Gemini API Key:**
    Create a file named `.env` in the root directory and add your API key:
    ```
    GEMINI_API_KEY="your_actual_api_key"
    ```

### Running the Application

The application will be available at `http://127.0.0.1:8000`

## API Endpoints

*   `POST /full-process`: Kicks off the initial processing of the entire message log.
*   `POST /process-next-batch`: Processes new messages since the last run.
*   `GET /solutions`: Retrieves the processed solutions as a JSON object.
*   `GET /markdown-report`: Generates a Markdown report based on the solution dictionary and returns it in the response, and saves the report to file named `solutions_report.md` at the `SAVE_PATH` variable specified in config.yaml.
## Configuration

Application settings like file paths and processing intervals can be modified in `configs/config.yaml`.
