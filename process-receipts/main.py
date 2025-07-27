import functions_framework
from google.cloud import bigquery
from google.cloud import storage
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
bq_client = bigquery.Client()
storage_client = storage.Client()

# Configuration
PROJECT_ID = "your-project-id"  # Update with your project ID
DATASET_ID = "my_dataset"
PARSED_TABLE = "invoices.parsed_user_receipts"
CATEGORIES_TABLE = "invoices.user_receipts_gcs"


@functions_framework.cloud_event
def process_document(cloud_event):
    """
    Cloud Run function triggered by Cloud Storage events.
    Processes documents using Document AI and stores results in BigQuery.
    """
    try:
        # Extract event data
        data = cloud_event.data
        bucket_name = data["bucket"]
        file_name = data["name"]
        content_type = data.get("contentType", "")

        # Construct full GCS URI
        gcs_uri = f"gs://{bucket_name}/{file_name}"

        logger.info(f"Processing file: {gcs_uri}")

        # Skip if not image or PDF
        if not _is_supported_file_type(content_type):
            logger.info(f"Skipping unsupported file type: {content_type}")
            return

        # Process document with Document AI
        parsed_data = _process_with_document_ai(gcs_uri, content_type)

        if parsed_data:
            # Store parsed data
            _store_parsed_data(parsed_data, gcs_uri)

            # Extract and store top categories
            _store_top_categories(parsed_data, gcs_uri)

            logger.info(f"Successfully processed: {gcs_uri}")
        else:
            logger.warning(f"No data extracted from: {gcs_uri}")

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise


def _is_supported_file_type(content_type):
    """Check if file type is supported (PDF or image)."""
    supported_types = [
        "application/pdf",
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/gif",
        "image/bmp",
        "image/tiff"
    ]
    return content_type.lower() in supported_types


def _process_with_document_ai(gcs_uri, content_type):
    """
    Process document using Document AI through BigQuery ML.
    Returns parsed document data.
    """
    try:
        # Create temporary table with document reference
        temp_table_id = f"{PROJECT_ID}.{DATASET_ID}.temp_receipt_{_generate_temp_id()}"

        # Create temp table with document URI
        create_temp_query = f"""
        CREATE OR REPLACE TABLE `{temp_table_id}` AS
        SELECT 
            '{gcs_uri}' as uri,
            '{content_type}' as content_type
        """

        bq_client.query(create_temp_query).result()

        # Process document using ML.PROCESS_DOCUMENT
        process_query = f"""
        SELECT *
        FROM ML.PROCESS_DOCUMENT(
            MODEL `{DATASET_ID}.invoice_parser`,
            TABLE `{temp_table_id}`
        )
        WHERE content_type = '{content_type}'
        """

        query_job = bq_client.query(process_query)
        results = query_job.result()

        # Convert results to list of dictionaries
        parsed_data = [dict(row) for row in results]

        # Clean up temp table
        bq_client.delete_table(temp_table_id)

        return parsed_data

    except Exception as e:
        logger.error(f"Error in Document AI processing: {str(e)}")
        return None


def _store_parsed_data(parsed_data, gcs_uri):
    """Store parsed document data in BigQuery."""
    try:
        table_ref = bq_client.get_table(PARSED_TABLE)

        # Add GCS URI to each record
        for record in parsed_data:
            record['source_uri'] = gcs_uri
            record['processed_timestamp'] = bq_client.query(
                "SELECT CURRENT_TIMESTAMP()").result().__next__()[0]

        # Insert data
        errors = bq_client.insert_rows_json(table_ref, parsed_data)

        if errors:
            logger.error(f"Errors inserting parsed data: {errors}")
        else:
            logger.info(f"Inserted {len(parsed_data)} parsed records")

    except Exception as e:
        logger.error(f"Error storing parsed data: {str(e)}")


def _store_top_categories(parsed_data, gcs_uri):
    """
    Extract top 5 categories by total_amount and store in user_receipts_gcs table.
    """
    try:
        # Extract user_id (assuming it's in the parsed data)
        user_id = _extract_user_id(parsed_data)

        if not user_id:
            logger.warning("No user_id found in parsed data")
            return

        # Aggregate categories by total_amount
        categories = _aggregate_categories(parsed_data)

        # Get top 5 categories
        top_categories = sorted(
            categories.items(), key=lambda x: x[1], reverse=True)[:5]

        if top_categories:
            # Prepare data for insertion
            category_record = {
                'user_id': user_id,
                'uri': gcs_uri,
                'top_categories': json.dumps([
                    {'category': cat, 'total_amount': float(amount)}
                    for cat, amount in top_categories
                ]),
                'processed_timestamp': bq_client.query("SELECT CURRENT_TIMESTAMP()").result().__next__()[0]
            }

            # Insert into categories table
            table_ref = bq_client.get_table(CATEGORIES_TABLE)
            errors = bq_client.insert_rows_json(table_ref, [category_record])

            if errors:
                logger.error(f"Errors inserting category data: {errors}")
            else:
                logger.info(
                    f"Stored top {len(top_categories)} categories for user {user_id}")

    except Exception as e:
        logger.error(f"Error storing top categories: {str(e)}")


def _extract_user_id(parsed_data):
    """Extract user_id from parsed data. Implement based on your data structure."""
    for record in parsed_data:
        if 'user_id' in record:
            return record['user_id']
        # Add other logic to extract user_id based on your document structure
    return None


def _aggregate_categories(parsed_data):
    """Aggregate line items by category and sum total amounts."""
    categories = {}

    for record in parsed_data:
        # Assuming parsed data contains line items with category and amount
        # Adjust field names based on your Document AI processor output
        if 'line_items' in record:
            for item in record['line_items']:
                category = item.get('category', 'Other')
                amount = float(item.get('amount', 0) or 0)

                if category in categories:
                    categories[category] += amount
                else:
                    categories[category] = amount

        # Alternative: if categories are at record level
        elif 'category' in record and 'total_amount' in record:
            category = record['category']
            amount = float(record['total_amount'] or 0)

            if category in categories:
                categories[category] += amount
            else:
                categories[category] = amount


def _generate_temp_id():
    """Generate unique ID for temporary table."""
    import uuid
    return str(uuid.uuid4()).replace('-', '_')
