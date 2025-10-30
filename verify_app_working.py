#!/usr/bin/env python3
"""
Verification script to check if the Smart Waste Management app is working
Tests the app by making HTTP requests to verify it's responding
"""

import requests
import time
import sys

def check_app_status():
    """Check if the Streamlit app is responding"""
    print("🔍 Checking Smart Waste Management App Status...")
    print("=" * 50)
    
    # App URL
    app_url = "http://localhost:8501"
    
    try:
        print(f"📡 Testing connection to {app_url}...")
        
        # Make a request to the app
        response = requests.get(app_url, timeout=10)
        
        if response.status_code == 200:
            print("✅ App is responding successfully!")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response Size: {len(response.content)} bytes")
            
            # Check if it contains expected content
            content = response.text.lower()
            
            expected_elements = [
                "smart waste management",
                "streamlit",
                "dashboard"
            ]
            
            found_elements = []
            for element in expected_elements:
                if element in content:
                    found_elements.append(element)
                    print(f"   ✅ Found: '{element}'")
                else:
                    print(f"   ⚠️ Missing: '{element}'")
            
            if len(found_elements) >= 2:
                print("\n🎉 SUCCESS: App is working correctly!")
                print("   The Smart Waste Management System is operational.")
                return True
            else:
                print("\n⚠️ WARNING: App is responding but may have issues.")
                return False
                
        else:
            print(f"❌ App returned error status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - App is not running or not accessible")
        print("   Make sure to start the app with: python -m streamlit run app.py")
        return False
        
    except requests.exceptions.Timeout:
        print("❌ Connection timed out - App may be starting up")
        print("   Wait a few seconds and try again")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_app_health_endpoints():
    """Check specific app health indicators"""
    print("\n🏥 Checking App Health Indicators...")
    
    # These would be actual health check endpoints in a production app
    health_checks = [
        ("Main Page", "http://localhost:8501"),
        ("Health Check", "http://localhost:8501/healthz"),  # May not exist
    ]
    
    results = []
    
    for name, url in health_checks:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"   ✅ {name}: OK")
                results.append(True)
            else:
                print(f"   ⚠️ {name}: Status {response.status_code}")
                results.append(False)
        except:
            print(f"   ❌ {name}: Not accessible")
            results.append(False)
    
    return any(results)

def main():
    """Main verification function"""
    print("🧪 Smart Waste Management App Verification")
    print("=" * 50)
    
    # Wait a moment for app to fully start
    print("⏳ Waiting for app to initialize...")
    time.sleep(3)
    
    # Check basic app status
    app_working = check_app_status()
    
    # Check health endpoints
    health_ok = check_app_health_endpoints()
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 Verification Summary:")
    
    if app_working:
        print("✅ App Status: WORKING")
        print("🌐 Access URL: http://localhost:8501")
        print("📱 Network URL: http://192.168.1.112:8501")
        
        print("\n🎯 What you can do now:")
        print("   1. Open http://localhost:8501 in your browser")
        print("   2. Login with demo credentials:")
        print("      - Email: amit.sharma.a@bmc.gov.in")
        print("      - Password: amitA@123")
        print("   3. Explore the real-time dashboard features")
        print("   4. Check system status and AI agents")
        print("   5. View live bin data and alerts")
        
        return True
    else:
        print("❌ App Status: NOT WORKING")
        print("\n🔧 Troubleshooting steps:")
        print("   1. Make sure the app is running: python -m streamlit run app.py")
        print("   2. Check for any error messages in the terminal")
        print("   3. Verify all dependencies are installed")
        print("   4. Try restarting the app")
        
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Verification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        sys.exit(1)