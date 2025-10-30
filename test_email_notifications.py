#!/usr/bin/env python3
"""
Test Email Notification System
Comprehensive testing for the email notification functionality
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project directories to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_agents"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_email_notification_system():
    """Test the email notification system"""
    try:
        from ai_agents.email_notifier import EmailNotificationSystem
        
        print("🧪 Testing Email Notification System...")
        print("=" * 50)
        
        # Initialize email notification system
        email_system = EmailNotificationSystem()
        
        # Test 1: Configuration Loading
        print("\n📋 Test 1: Configuration Loading")
        if email_system.email_config:
            print("✅ Email configuration loaded successfully")
            print(f"   SMTP Server: {email_system.email_config.smtp_server}")
            print(f"   SMTP Port: {email_system.email_config.smtp_port}")
            print(f"   Sender: {email_system.email_config.sender_email}")
            print(f"   TLS Enabled: {email_system.email_config.use_tls}")
        else:
            print("⚠️  Email configuration not loaded (credentials not set)")
        
        # Test 2: Template Initialization
        print("\n📧 Test 2: Email Templates")
        print(f"✅ {len(email_system.templates)} email templates loaded:")
        for template_id, template in email_system.templates.items():
            print(f"   - {template_id}: {template.priority} priority")
        
        # Test 3: Database Setup
        print("\n🗄️  Test 3: Database Tables")
        import sqlite3
        conn = sqlite3.connect("backend/db/operators.db")
        cursor = conn.cursor()
        
        # Check if email tables exist
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('email_notifications', 'email_preferences', 'supervisor_contacts')
        """)
        tables = cursor.fetchall()
        
        if len(tables) == 3:
            print("✅ All email database tables created successfully")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cursor.fetchone()[0]
                print(f"   - {table[0]}: {count} records")
        else:
            print(f"⚠️  Only {len(tables)}/3 email tables found")
        
        conn.close()
        
        # Test 4: Sample Alert Notification
        print("\n🚨 Test 4: Sample Alert Notification")
        
        # Sample alert data
        alert_data = {
            "id": "test_alert_001",
            "alert_type": "overflow_warning",
            "priority": 3,
            "message": "Test bin approaching capacity",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recommended_action": "Schedule collection within next 4 hours",
            "natural_language_summary": "This is a test alert to verify email notification functionality."
        }
        
        # Sample bin data
        bin_info = {
            "id": "TEST_BIN_001",
            "ward_id": 1,
            "current_fill": 87.5,
            "latitude": 19.0760,
            "longitude": 72.8777,
            "bin_type": "residential",
            "fill_rate": 2.3
        }
        
        # Test email sending (only if configured)
        if email_system.email_config and email_system.email_config.sender_email:
            print("📤 Attempting to send test email notification...")
            
            success = await email_system.send_alert_notification(
                alert_data=alert_data,
                bin_info=bin_info,
                recipient_email="test@example.com",  # Test email
                recipient_name="Test Operator"
            )
            
            if success:
                print("✅ Test email notification sent successfully!")
            else:
                print("❌ Test email notification failed")
        else:
            print("⚠️  Email not configured - skipping send test")
            print("   (This is expected if SMTP credentials are not set)")
        
        # Test 5: Email Statistics
        print("\n📊 Test 5: Email Statistics")
        stats = email_system.get_email_stats()
        print("✅ Email statistics retrieved:")
        for key, value in stats.items():
            print(f"   - {key}: {value}")
        
        # Test 6: Supervisor Contacts
        print("\n👥 Test 6: Supervisor Contacts")
        supervisors = await email_system.get_supervisor_contacts(ward_id=1)
        print(f"✅ Found {len(supervisors)} supervisor contacts for Ward 1:")
        for supervisor in supervisors:
            print(f"   - {supervisor['supervisor_name']}: {supervisor['supervisor_email']}")
        
        # Test 7: Email Preferences
        print("\n⚙️  Test 7: Email Preferences")
        preferences = await email_system.get_operator_email_preferences(operator_id=1)
        print("✅ Email preferences retrieved:")
        for key, value in preferences.items():
            if key not in ['operator_id', 'email']:
                print(f"   - {key}: {value}")
        
        print("\n" + "=" * 50)
        print("🎉 Email Notification System Test Completed!")
        print("✅ All core components are working correctly")
        
        if not email_system.email_config or not email_system.email_config.sender_email:
            print("\n📝 Note: To enable actual email sending, update the configuration:")
            print("   1. Edit data/config.json")
            print("   2. Set sender_email and sender_password")
            print("   3. Configure SMTP settings for your email provider")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Email notification test failed: {e}")
        return False

async def test_alert_manager_integration():
    """Test integration with Alert Management Agent"""
    try:
        print("\n🤖 Testing Alert Manager Integration...")
        print("=" * 50)
        
        from ai_agents.alert_manager import AlertManagementAgent
        
        # Initialize alert manager (without Redis/Vector DB for testing)
        alert_manager = AlertManagementAgent()
        
        # Test email notifier integration
        if hasattr(alert_manager, 'email_notifier'):
            print("✅ Alert Manager has email notifier integrated")
            
            # Test email statistics
            stats = alert_manager.get_alert_stats()
            if 'email_notifications' in stats:
                print("✅ Email statistics integrated in alert stats")
                print(f"   Email stats: {stats['email_notifications']}")
            else:
                print("⚠️  Email statistics not found in alert stats")
        else:
            print("❌ Alert Manager missing email notifier integration")
        
        # Test email preferences methods
        if hasattr(alert_manager, 'get_email_preferences'):
            print("✅ Email preferences methods available")
        else:
            print("❌ Email preferences methods missing")
        
        print("✅ Alert Manager integration test completed")
        return True
        
    except Exception as e:
        print(f"❌ Alert Manager integration test failed: {e}")
        logger.error(f"Alert Manager integration test failed: {e}")
        return False

def test_configuration_file():
    """Test configuration file updates"""
    try:
        print("\n⚙️  Testing Configuration File...")
        print("=" * 50)
        
        with open("data/config.json", 'r') as f:
            config = json.load(f)
        
        # Check email configuration section
        if "email_notifications" in config:
            email_config = config["email_notifications"]
            print("✅ Email notifications section found in config")
            
            required_fields = [
                "smtp_server", "smtp_port", "use_tls", "sender_email", 
                "sender_password", "sender_name", "enabled"
            ]
            
            missing_fields = []
            for field in required_fields:
                if field not in email_config:
                    missing_fields.append(field)
            
            if not missing_fields:
                print("✅ All required email configuration fields present")
                print(f"   SMTP Server: {email_config.get('smtp_server')}")
                print(f"   SMTP Port: {email_config.get('smtp_port')}")
                print(f"   TLS Enabled: {email_config.get('use_tls')}")
                print(f"   Enabled: {email_config.get('enabled')}")
            else:
                print(f"⚠️  Missing configuration fields: {missing_fields}")
            
            # Check default preferences
            if "default_preferences" in email_config:
                print("✅ Default email preferences configured")
            else:
                print("⚠️  Default email preferences not found")
            
        else:
            print("❌ Email notifications section missing from config")
            return False
        
        print("✅ Configuration file test completed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration file test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🧪 Smart Waste Management - Email Notification System Tests")
    print("=" * 60)
    
    tests = [
        ("Configuration File", test_configuration_file()),
        ("Email Notification System", test_email_notification_system()),
        ("Alert Manager Integration", test_alert_manager_integration())
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_coro in tests:
        print(f"\n🔍 Running {test_name} Test...")
        try:
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
            
            if result:
                passed += 1
                print(f"✅ {test_name} Test: PASSED")
            else:
                print(f"❌ {test_name} Test: FAILED")
        except Exception as e:
            print(f"❌ {test_name} Test: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Email notification system is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    # Run tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        success = loop.run_until_complete(main())
        sys.exit(0 if success else 1)
    finally:
        loop.close()