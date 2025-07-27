#!/bin/bash

# Invoice Multi-Agent Deployment Script for Google Cloud
# This script provides multiple deployment options for the invoice processing service

set -e

# Configuration
PROJECT_ID="chitgenie"
REGION="us-central1"
SERVICE_NAME="invoice-agent"
GEMINI_API_KEY="${GEMINI_API_KEY:-AIzaSyCuI8HwBRLXudqEeOSoksvfMlvyuC314XY}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Invoice Multi-Agent Deployment${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check if user is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null; then
        print_error "Please authenticate with gcloud: gcloud auth login"
        exit 1
    fi
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    print_status "Prerequisites check passed!"
}

enable_apis() {
    print_status "Enabling required Google Cloud APIs..."
    
    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        containerregistry.googleapis.com \
        bigquery.googleapis.com \
        aiplatform.googleapis.com \
        appengine.googleapis.com
    
    print_status "APIs enabled successfully!"
}

setup_bigquery() {
    print_status "Setting up BigQuery dataset..."
    
    # Create dataset if it doesn't exist
    bq mk --dataset --location=US --description="Expense tracking dataset" \
        $PROJECT_ID:expense_tracking 2>/dev/null || true
    
    print_status "BigQuery setup completed!"
}

deploy_cloud_run() {
    print_status "Deploying to Cloud Run..."
    
    # Build and deploy using Cloud Build
    gcloud builds submit --config cloudbuild.yaml --substitutions _PROJECT_ID=$PROJECT_ID
    
    # Update service with environment variables
    gcloud run services update $SERVICE_NAME \
        --region=$REGION \
        --set-env-vars="GEMINI_API_KEY=$GEMINI_API_KEY,GCP_PROJECT_ID=$PROJECT_ID" \
        --quiet
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
    
    print_status "Cloud Run deployment completed!"
    print_status "Service URL: $SERVICE_URL"
    print_status "Health check: curl $SERVICE_URL/health"
}

deploy_app_engine() {
    print_status "Deploying to App Engine..."
    
    # Create app.yaml with API key
    cat > app.yaml << EOF
runtime: python311

instance_class: F4_1G
automatic_scaling:
  min_instances: 0
  max_instances: 10
  target_cpu_utilization: 0.6

env_variables:
  GCP_PROJECT_ID: "$PROJECT_ID"
  GEMINI_API_KEY: "$GEMINI_API_KEY"

resources:
  cpu: 2
  memory_gb: 2

health_check:
  enable_health_check: true
  check_interval_sec: 30
  timeout_sec: 4
EOF
    
    # Deploy to App Engine
    gcloud app deploy app.yaml --quiet
    
    # Get service URL
    SERVICE_URL=$(gcloud app describe --format="value(defaultHostname)")
    SERVICE_URL="https://$SERVICE_URL"
    
    print_status "App Engine deployment completed!"
    print_status "Service URL: $SERVICE_URL"
    print_status "Health check: curl $SERVICE_URL/health"
}

local_docker_run() {
    print_status "Building and running locally with Docker..."
    
    # Build Docker image
    docker build -t invoice-agent:local .
    
    # Run container
    docker run -d \
        -p 8080:8080 \
        -e GEMINI_API_KEY="$GEMINI_API_KEY" \
        -e GCP_PROJECT_ID="$PROJECT_ID" \
        -e GOOGLE_APPLICATION_CREDENTIALS="/app/credentials.json" \
        -v "$HOME/.config/gcloud/application_default_credentials.json:/app/credentials.json:ro" \
        --name invoice-agent-local \
        invoice-agent:local
    
    print_status "Local Docker deployment completed!"
    print_status "Service URL: http://localhost:8080"
    print_status "Health check: curl http://localhost:8080/health"
    print_status "Stop with: docker stop invoice-agent-local"
    print_status "Remove with: docker rm invoice-agent-local"
}

test_deployment() {
    local service_url=$1
    
    print_status "Testing deployment..."
    
    # Test health endpoint
    if curl -f "$service_url/health" > /dev/null 2>&1; then
        print_status "Health check passed!"
    else
        print_error "Health check failed!"
        return 1
    fi
    
    # Test categories endpoint
    if curl -f "$service_url/categories" > /dev/null 2>&1; then
        print_status "Categories endpoint working!"
    else
        print_warning "Categories endpoint may have issues"
    fi
    
    # Test query endpoint
    if curl -f -X POST "$service_url/query" \
        -H "Content-Type: application/json" \
        -d '{"query": "test query", "user_id": "test-user"}' > /dev/null 2>&1; then
        print_status "Query endpoint working!"
    else
        print_warning "Query endpoint may have issues"
    fi
    
    print_status "Basic tests completed!"
}

show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --cloud-run     Deploy to Google Cloud Run (recommended)"
    echo "  --app-engine    Deploy to Google App Engine"
    echo "  --local         Run locally with Docker"
    echo "  --setup-only    Only setup prerequisites and BigQuery"
    echo "  --test URL      Test deployment at given URL"
    echo "  --help          Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  GEMINI_API_KEY  Your Gemini API key (required)"
    echo "  PROJECT_ID      GCP Project ID (default: chitgenie)"
    echo ""
    echo "Examples:"
    echo "  $0 --cloud-run"
    echo "  $0 --local"
    echo "  $0 --test https://your-service-url.com"
}

# Main execution
main() {
    print_header
    
    case "${1:-}" in
        --cloud-run)
            check_prerequisites
            enable_apis
            setup_bigquery
            deploy_cloud_run
            ;;
        --app-engine)
            check_prerequisites
            enable_apis
            setup_bigquery
            deploy_app_engine
            ;;
        --local)
            local_docker_run
            ;;
        --setup-only)
            check_prerequisites
            enable_apis
            setup_bigquery
            print_status "Setup completed! You can now deploy using --cloud-run or --app-engine"
            ;;
        --test)
            if [ -z "$2" ]; then
                print_error "Please provide service URL for testing"
                show_usage
                exit 1
            fi
            test_deployment "$2"
            ;;
        --help|"")
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ] && [[ "$1" != "--help" ]] && [[ "$1" != "" ]]; then
    print_error "GEMINI_API_KEY environment variable is required"
    print_status "Set it with: export GEMINI_API_KEY='your-api-key'"
    exit 1
fi

main "$@"