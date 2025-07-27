#!/usr/bin/env python3
"""
Test script for Invoice Multi-Agent REST API
"""

import requests
import json
import base64
import time
from typing import Dict, Any

class InvoiceAgentTester:
    def __init__(self, base_url: str):
        """Initialize the tester with the API base URL"""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'InvoiceAgent-Tester/1.0'
        })
    
    def test_health(self) -> bool:
        """Test the health endpoint"""
        print("🔍 Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data.get('message', 'OK')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_categories(self) -> bool:
        """Test the categories endpoint"""
        print("\n🔍 Testing categories endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/categories")
            if response.status_code == 200:
                data = response.json()
                categories = data.get('categories', [])
                synonyms = data.get('category_synonyms', {})
                print(f"✅ Categories retrieved: {len(categories)} categories")
                print(f"📋 Available categories: {', '.join(categories[:5])}...")
                print(f"🔗 Synonyms available: {len(synonyms)} mappings")
                return True
            else:
                print(f"❌ Categories failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Categories error: {e}")
            return False
    
    def test_query(self, query: str, user_id: str = "TEST-USER") -> bool:
        """Test natural language query processing"""
        print(f"\n🔍 Testing query: '{query}'")
        try:
            payload = {
                "query": query,
                "user_id": user_id
            }
            response = self.session.post(f"{self.base_url}/query", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✅ Query processed successfully")
                    print(f"💬 Response: {data.get('response', 'No response')[:100]}...")
                    return True
                else:
                    print(f"❌ Query failed: {data.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Query HTTP error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error details: {error_data.get('error', 'No details')}")
                except:
                    print(f"   Response: {response.text[:200]}")
                return False
        except Exception as e:
            print(f"❌ Query error: {e}")
            return False
    
    def test_expenses(self, user_id: str = "TEST-USER", days: int = 30) -> bool:
        """Test expenses retrieval"""
        print(f"\n🔍 Testing expenses for user {user_id} (last {days} days)...")
        try:
            response = self.session.get(f"{self.base_url}/expenses/{user_id}?days={days}&limit=5")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    expenses = data.get('expenses', [])
                    count = data.get('count', 0)
                    print(f"✅ Expenses retrieved: {count} records")
                    if expenses:
                        first_expense = expenses[0]
                        print(f"📊 Sample expense: {first_expense.get('vendor_name')} - ₹{first_expense.get('amount', 0)}")
                    return True
                else:
                    print(f"❌ Expenses failed: {data.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Expenses HTTP error: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Expenses error: {e}")
            return False
    
    def test_summary(self, user_id: str = "TEST-USER", days: int = 30) -> bool:
        """Test expense summary"""
        print(f"\n🔍 Testing summary for user {user_id} (last {days} days)...")
        try:
            response = self.session.get(f"{self.base_url}/summary/{user_id}?days={days}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    total_spent = data.get('total_spent', 0)
                    total_transactions = data.get('total_transactions', 0)
                    categories = data.get('categories', [])
                    print(f"✅ Summary retrieved successfully")
                    print(f"💰 Total spent: ₹{total_spent:,.2f}")
                    print(f"📊 Total transactions: {total_transactions}")
                    print(f"📋 Categories: {len(categories)}")
                    return True
                else:
                    print(f"❌ Summary failed: {data.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Summary HTTP error: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Summary error: {e}")
            return False
    
    def test_invoice_base64(self, sample_image_b64: str, user_id: str = "TEST-USER") -> bool:
        """Test invoice processing with base64 image"""
        print(f"\n🔍 Testing invoice upload (base64) for user {user_id}...")
        try:
            payload = {
                "image": sample_image_b64,
                "user_id": user_id
            }
            
            # Remove Content-Type for this request to let requests set it
            headers = {'User-Agent': 'InvoiceAgent-Tester/1.0'}
            response = requests.post(f"{self.base_url}/upload-invoice-base64", 
                                   json=payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    extracted = data.get('extracted_data', {})
                    print(f"✅ Invoice processed successfully")
                    print(f"🏪 Vendor: {extracted.get('vendor_name', 'Unknown')}")
                    print(f"💰 Amount: ₹{extracted.get('amount', 0)}")
                    print(f"📂 Category: {extracted.get('category', 'Unknown')}")
                    print(f"🎯 Confidence: {extracted.get('confidence', 0):.0%}")
                    return True
                else:
                    print(f"❌ Invoice processing failed: {data.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Invoice HTTP error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error details: {error_data.get('error', 'No details')}")
                except:
                    print(f"   Response: {response.text[:200]}")
                return False
        except Exception as e:
            print(f"❌ Invoice error: {e}")
            return False
    
    def run_comprehensive_test(self, user_id: str = "TEST-USER") -> Dict[str, bool]:
        """Run comprehensive test suite"""
        print("🚀 Starting comprehensive API test suite...")
        print("=" * 60)
        
        results = {}
        
        # Test basic endpoints
        results['health'] = self.test_health()
        results['categories'] = self.test_categories()
        
        # Test query processing with various questions
        test_queries = [
            "How much did I spend on groceries last month?",
            "Show me my electricity bills",
            "Can I return clothes I bought recently?",
            "When is my insurance due?",
            "What are my spending trends?",
            "How much do I spend on average per month?",
        ]
        
        query_results = []
        for query in test_queries:
            result = self.test_query(query, user_id)
            query_results.append(result)
            time.sleep(1)  # Rate limiting courtesy
        
        results['queries'] = all(query_results)
        
        # Test data retrieval
        results['expenses'] = self.test_expenses(user_id)
        results['summary'] = self.test_summary(user_id)
        
        # Test invoice processing with a minimal base64 image
        # This is a 1x1 pixel white JPEG image for testing
        sample_image = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDAREAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A/9k="
        results['invoice'] = self.test_invoice_base64(f"data:image/jpeg;base64,{sample_image}", user_id)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Test Results Summary:")
        print("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"{test_name.upper():15} : {status}")
        
        print("-" * 60)
        print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! Your API is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the logs above for details.")
        
        return results

def main():
    """Main function to run tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Invoice Multi-Agent REST API')
    parser.add_argument('--url', required=True, help='Base URL of the API (e.g., https://your-service.com)')
    parser.add_argument('--user-id', default='TEST-USER', help='User ID for testing (default: TEST-USER)')
    parser.add_argument('--query', help='Test a specific query')
    parser.add_argument('--quick', action='store_true', help='Run only basic tests')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = InvoiceAgentTester(args.url)
    
    if args.query:
        # Test specific query
        print(f"Testing specific query with {args.url}")
        success = tester.test_query(args.query, args.user_id)
        exit(0 if success else 1)
    elif args.quick:
        # Quick test
        print(f"Running quick tests with {args.url}")
        health_ok = tester.test_health()
        categories_ok = tester.test_categories()
        query_ok = tester.test_query("How much did I spend this month?", args.user_id)
        
        if health_ok and categories_ok and query_ok:
            print("✅ Quick tests passed!")
            exit(0)
        else:
            print("❌ Quick tests failed!")
            exit(1)
    else:
        # Comprehensive test
        print(f"Running comprehensive tests with {args.url}")
        results = tester.run_comprehensive_test(args.user_id)
        
        # Exit with appropriate code
        exit(0 if all(results.values()) else 1)

if __name__ == "__main__":
    main()