# Receipt Processing System

## Overview

An intelligent receipt processing system that handles user-submitted receipts through WhatsApp/Telegram/SMS, extracts data using OCR, categorizes expenses, and provides business intelligence insights. The system features both user-facing conversational interfaces and business-facing analytics.

## Architecture

### Core Components

#### Inbound Processing
- **Input Channels**: WhatsApp, Telegram, SMS
- **Media Handler**: Processes images, documents, videos
- **OCR Engine**: Extracts text and data from receipt images
- **Cloud Storage**: Google Cloud Storage for media persistence

#### Data Processing Pipeline
- **BigQuery Integration**: 
  - Table 1: `parsed_user_receipts` - Raw receipt data
  - Table 2: `parsed_user_receipts_structured` - Processed invoice data with amounts, dates, categories
  - Table 3: `UserExpenditureStats` - Aggregated spending analytics

#### Intelligent Processing
- **Google Cloud Functions**: 
  - Receipt validation and confirmation
  - Contact data parsing
  - Expense categorization using ML
  - Invoice parsing and amount extraction

#### User Interface
- **Agentic Core (Vertex AI)**:
  - Intent classification
  - Expenditure analysis
  - Reminder requests
  - Response generation
- **Multi-channel Support**:
  - WhatsApp/Telegram messaging
  - Google Calendar integration
  - GenAI conversational responses
  - Google Vision API for advanced OCR

#### Business Intelligence
- **Analytics Dashboard**: 
  - Query-based search functionality
  - Statistical consumption analysis
- **Business User Interface**: Dedicated portal for business insights

## Features

### User Features
- **Multi-Platform Support**: Submit receipts via WhatsApp, Telegram, or SMS
- **Intelligent OCR**: Automatic text extraction from receipt images
- **Smart Categorization**: ML-powered expense classification
- **Conversational Interface**: Natural language queries about spending
- **Calendar Integration**: Expense reminders and scheduling
- **Spending Analytics**: Personal expenditure insights

### Business Features
- **Advanced Search**: Query-based receipt and expense searching
- **Statistical Analysis**: Consumption patterns and trends
- **User Behavior Insights**: Aggregate spending analytics
- **Data Export**: Structured data access via BigQuery

## Technology Stack

### Cloud Infrastructure
- **Google Cloud Platform**
  - BigQuery (Data Warehouse)
  - Cloud Functions (Serverless Processing)
  - Cloud Storage (Media Storage)
  - Vertex AI (ML/AI Processing)
  - Vision API (OCR)

### Communication Channels
- WhatsApp Business API
- Telegram Bot API
- SMS Gateway Integration

### Data Processing
- **OCR**: Google Vision API + custom processing
- **ML Classification**: Vertex AI for categorization
- **Data Pipeline**: Cloud Functions → BigQuery

## Data Flow

1. **Receipt Submission**: User sends receipt via messaging platform
2. **Media Processing**: System stores image and triggers OCR
3. **Data Extraction**: OCR extracts text, amounts, dates, merchant info
4. **Validation**: Cloud Function confirms receipt validity
5. **Categorization**: ML model classifies expense type
6. **Storage**: Structured data stored in BigQuery tables
7. **Analytics**: Data aggregated for user and business insights
8. **Response**: User receives confirmation and can query data

## Database Schema

### Table 1: `parsed_user_receipts`
```sql
- user_id (STRING)
- receipt_id (STRING)
- raw_text (STRING)
- image_url (STRING)
- timestamp (TIMESTAMP)
- file_type (STRING)
```

### Table 2: `parsed_user_receipts_structured`
```sql
- user_id (STRING)
- receipt_id (STRING)
- invoice_date (DATE)
- total_amount (FLOAT)
- merchant_name (STRING)
- category (STRING)
- items (ARRAY<STRUCT>)
```

### Table 3: `UserExpenditureStats`
```sql
- user_id (STRING)
- period (STRING)
- category (STRING)
- total_spent (FLOAT)
- transaction_count (INTEGER)
- avg_transaction (FLOAT)
```

## Setup and Deployment

### Prerequisites
- Google Cloud Platform account
- WhatsApp Business API access
- Telegram Bot Token
- SMS gateway credentials

### Environment Variables
```bash
GCP_PROJECT_ID=your-project-id
BIGQUERY_DATASET=receipt_processing
WHATSAPP_TOKEN=your-whatsapp-token
TELEGRAM_TOKEN=your-telegram-token
VERTEX_AI_REGION=us-central1
```

### Deployment Steps
1. Set up GCP project and enable required APIs
2. Deploy Cloud Functions for processing pipeline
3. Configure BigQuery datasets and tables
4. Set up messaging platform webhooks
5. Deploy Vertex AI models for classification
6. Configure business dashboard

