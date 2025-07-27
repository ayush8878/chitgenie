#!/bin/bash

# Exit on any error
set -e

echo "🚀 Starting deployment of Gemini BigQuery Bot..."

# Navigate to the functions directory
cd "$(dirname "$0")/.."

# Check if firebase-tools is installed
if ! command -v firebase &> /dev/null; then
    echo "❌ Firebase CLI is not installed. Please install it with 'npm install -g firebase-tools'"
    exit 1
fi

# Check if user is logged in to Firebase
if ! firebase projects:list 2>&1 | grep -q "$(firebase use 2>&1 | grep 'project' | awk '{print $2}' | tr -d '[:space:]')"; then
    echo "🔐 Logging in to Firebase..."
    firebase login
fi

# Set the project (you'll need to replace this with your Firebase project ID)
# firebase use your-project-id

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Lint and test (uncomment these lines if you have tests)
# echo "🔍 Linting..."
# npm run lint

# echo "🧪 Running tests..."
# npm test

# Deploy to Firebase
echo "🚀 Deploying to Firebase..."
firebase deploy --only functions:queryBigQuery

echo "✅ Deployment complete!"
echo "🔗 Your function is now live at: $(firebase functions:list | grep queryBigQuery | awk '{print $2}')"
