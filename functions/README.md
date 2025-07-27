# Gemini BigQuery Bot - Firebase Function

This project exposes the GeminiBigQueryBot as a Firebase Cloud Function, allowing you to query BigQuery using natural language.

## Prerequisites

1. Node.js 18 or higher
2. Firebase CLI (`npm install -g firebase-tools`)
3. Google Cloud Project with:
   - BigQuery API enabled
   - A service account with BigQuery access
   - Gemini API enabled

## Setup

1. **Install dependencies**:
   ```bash
   cd functions
   npm install
   ```

2. **Configure environment variables**:
   Create a `.env` file in the `functions` directory with the following variables:
   ```
   GOOGLE_API_KEY=your-gemini-api-key
   GOOGLE_CLOUD_PROJECT=your-gcp-project-id
   DEFAULT_PROJECT_ID=your-bigquery-project-id
   DEFAULT_DATASET_ID=your-dataset-id
   DEFAULT_TABLE_ID=your-table-id
   ```

3. **Login to Firebase**:
   ```bash
   firebase login
   ```

4. **Set your Firebase project**:
   ```bash
   firebase use your-firebase-project-id
   ```

## Local Development

To run the function locally:

```bash
# Start the emulator
npm run serve

# Or run in shell mode
npm start
```

The function will be available at `http://localhost:5001/your-project-id/us-central1/queryBigQuery`

## Deployment

1. **Deploy the function**:
   ```bash
   chmod +x scripts/deploy.sh
   ./scripts/deploy.sh
   ```

   Or manually:
   ```bash
   firebase deploy --only functions:queryBigQuery
   ```

2. **Test the deployed function**:
   ```bash
   curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"query": "Show me the top 10 highest invoices"}' \
     https://us-central1-your-project-id.cloudfunctions.net/queryBigQuery
   ```

## Usage

Send a POST request to the function with a JSON body containing:
- `query`: (required) Your natural language query
- `projectId`: (optional) Override default BigQuery project ID
- `datasetId`: (optional) Override default dataset ID
- `tableId`: (optional) Override default table ID

Example:
```json
{
  "query": "Show me all overdue invoices from the last 30 days",
  "projectId": "my-bigquery-project",
  "datasetId": "invoices",
  "tableId": "transactions"
}
```

## Security

- The function includes basic SQL injection protection
- Ensure your Firebase Security Rules are properly configured
- Use Firebase Authentication to secure your endpoint if needed
- Rotate your API keys regularly

## License

MIT
