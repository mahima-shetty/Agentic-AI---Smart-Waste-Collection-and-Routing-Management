# Email Notification System - Setup and Usage Guide

## Overview

The Smart Waste Management System now includes a comprehensive email notification system that automatically sends alerts for bin overflows, escalations, and system events. This guide explains how to configure and use the email notification features.

## Features Implemented

### ✅ Core Features
- **SMTP Integration**: Full SMTP support with TLS encryption
- **Email Templates**: Professional HTML templates for different alert types
- **Supervisor Escalation**: Automatic escalation emails to supervisors
- **Email Preferences**: Operator-configurable notification preferences
- **Delivery Confirmation**: Email delivery tracking and retry mechanism
- **Dashboard Integration**: Email preferences management in the dashboard

### ✅ Email Types
1. **Overflow Warning**: Sent when bins reach 85% capacity
2. **Critical Overflow**: Sent when bins reach 95% capacity
3. **Predictive Alerts**: AI-powered overflow predictions
4. **Escalation Notifications**: Multi-level supervisor escalation
5. **System Notifications**: General system alerts

## Configuration

### 1. Email Server Configuration

Edit `data/config.json` to configure your SMTP settings:

```json
{
  "email_notifications": {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "use_tls": true,
    "sender_email": "your-email@gmail.com",
    "sender_password": "your-app-password",
    "sender_name": "Smart Waste Management System - BMC Mumbai",
    "enabled": true
  }
}
```

### 2. Gmail Setup (Recommended)

For Gmail, you need to:
1. Enable 2-Factor Authentication
2. Generate an App Password
3. Use the App Password in the configuration

### 3. Other Email Providers

For other providers, update the SMTP settings accordingly:

**Outlook/Hotmail:**
```json
{
  "smtp_server": "smtp-mail.outlook.com",
  "smtp_port": 587,
  "use_tls": true
}
```

**Yahoo:**
```json
{
  "smtp_server": "smtp.mail.yahoo.com",
  "smtp_port": 587,
  "use_tls": true
}
```

## Usage

### 1. Automatic Notifications

The system automatically sends emails when:
- Bins reach warning levels (85% full)
- Bins reach critical levels (95% full)
- AI predicts overflow events
- Multiple critical alerts trigger escalation

### 2. Dashboard Email Preferences

Operators can configure their email preferences through the dashboard:

1. **Login** to the dashboard
2. **Click** "⚙️ Email Preferences" in the sidebar
3. **Configure** notification types:
   - Alert Notifications
   - Escalation Notifications
   - Daily Reports
   - Weekly Summaries
   - Emergency Alerts

4. **Set** notification timing:
   - Frequency (immediate, hourly, daily)
   - Quiet hours (no non-emergency notifications)

### 3. Testing Email System

Use the built-in test functionality:

```bash
# Run comprehensive email system tests
python test_email_notifications.py

# Test from dashboard
# Click "📧 Test Email" in the sidebar
```

## Email Templates

### Template Types

1. **Overflow Warning** (High Priority)
   - Professional orange-themed design
   - Bin details and recommended actions
   - Context-aware messaging

2. **Critical Overflow** (Urgent Priority)
   - Red-themed urgent design
   - Immediate action required
   - Escalation information

3. **Predictive Alert** (Normal Priority)
   - Blue-themed AI prediction design
   - Time-based predictions
   - Proactive recommendations

4. **Escalation** (Urgent Priority)
   - Pink-themed supervisor alert
   - Multiple bin information
   - Coordinated response suggestions

### Template Customization

Templates are defined in `ai_agents/email_notifier.py` and can be customized:

```python
# Example template modification
self.templates["overflow_warning"] = EmailTemplate(
    template_id="overflow_warning",
    subject_template="⚠️ Custom Subject - {bin_id}",
    body_template="<html>Custom HTML content...</html>",
    is_html=True,
    priority="high"
)
```

## Database Schema

### Email Notifications Table
```sql
CREATE TABLE email_notifications (
    notification_id TEXT PRIMARY KEY,
    recipient_email TEXT NOT NULL,
    template_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    delivery_status TEXT DEFAULT 'pending',
    created_at TEXT NOT NULL,
    sent_at TEXT,
    error_message TEXT
);
```

### Email Preferences Table
```sql
CREATE TABLE email_preferences (
    operator_id INTEGER PRIMARY KEY,
    email TEXT NOT NULL,
    alert_notifications BOOLEAN DEFAULT 1,
    escalation_notifications BOOLEAN DEFAULT 1,
    notification_frequency TEXT DEFAULT 'immediate',
    quiet_hours_start TEXT DEFAULT '22:00',
    quiet_hours_end TEXT DEFAULT '06:00'
);
```

### Supervisor Contacts Table
```sql
CREATE TABLE supervisor_contacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ward_id INTEGER NOT NULL,
    supervisor_name TEXT NOT NULL,
    supervisor_email TEXT NOT NULL,
    escalation_level INTEGER DEFAULT 1
);
```

## API Integration

### Alert Manager Integration

The email system is fully integrated with the Alert Management Agent:

```python
# Email notifications are automatically sent when alerts are generated
alert_manager = AlertManagementAgent()
await alert_manager.start_monitoring()  # Emails sent automatically

# Manual email preferences management
preferences = await alert_manager.get_email_preferences(operator_id=1)
await alert_manager.update_email_preferences(operator_id=1, preferences)
```

### Dashboard Integration

Email preferences are accessible through the enhanced dashboard:

```python
# Run enhanced dashboard with email features
from dashboard.enhanced_dashboard import run_enhanced_dashboard
run_enhanced_dashboard(system_initializer)
```

## Monitoring and Statistics

### Email Statistics

Monitor email performance through:

1. **Dashboard Analytics**: View email delivery rates and statistics
2. **System Status**: Check email system health
3. **Alert Stats**: Email notifications included in alert statistics

### Key Metrics

- **Total Sent**: Number of emails successfully sent
- **Delivery Rate**: Percentage of successful deliveries
- **Failed Count**: Number of failed email attempts
- **Pending Count**: Emails waiting to be sent
- **Average Response Time**: Email sending performance

## Troubleshooting

### Common Issues

1. **Authentication Failed**
   - Check email credentials
   - Verify app password for Gmail
   - Ensure 2FA is enabled

2. **Connection Timeout**
   - Verify SMTP server and port
   - Check firewall settings
   - Confirm TLS settings

3. **Emails Not Received**
   - Check spam/junk folders
   - Verify recipient email addresses
   - Review email preferences settings

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger('ai_agents.email_notifier').setLevel(logging.DEBUG)
```

### Test Commands

```bash
# Test email configuration
python test_email_notifications.py

# Test specific components
python -c "
import asyncio
from ai_agents.email_notifier import EmailNotificationSystem
email_system = EmailNotificationSystem()
print('Email system initialized:', email_system.email_config is not None)
"
```

## Security Considerations

### Best Practices

1. **Use App Passwords**: Never use main account passwords
2. **Enable TLS**: Always use encrypted connections
3. **Secure Storage**: Store credentials securely
4. **Access Control**: Limit email configuration access
5. **Regular Updates**: Keep email credentials updated

### Configuration Security

```json
{
  "email_notifications": {
    "sender_password": "use-app-password-not-main-password",
    "use_tls": true,
    "enabled": true
  }
}
```

## Performance Optimization

### Email Sending

- **Thread Pool**: Emails sent in background threads
- **Retry Logic**: Automatic retry for failed emails
- **Rate Limiting**: Prevents email server overload
- **Batch Processing**: Efficient handling of multiple emails

### Database Optimization

- **Indexed Queries**: Fast email history retrieval
- **Cleanup Tasks**: Automatic old email cleanup
- **Connection Pooling**: Efficient database connections

## Future Enhancements

### Planned Features

1. **Email Scheduling**: Schedule emails for optimal delivery times
2. **Rich Analytics**: Advanced email performance analytics
3. **Template Editor**: Visual email template customization
4. **Multi-language**: Support for multiple languages
5. **SMS Integration**: SMS notifications for critical alerts

## Support

For issues or questions:

1. **Check Logs**: Review `smart_waste_system.log`
2. **Run Tests**: Execute `test_email_notifications.py`
3. **Dashboard Status**: Check system status in dashboard
4. **Configuration**: Verify `data/config.json` settings

## Summary

The email notification system is now fully integrated and provides:

✅ **SMTP Integration** - Professional email delivery
✅ **HTML Templates** - Beautiful, responsive email designs  
✅ **Supervisor Escalation** - Multi-level alert escalation
✅ **Email Preferences** - User-configurable settings
✅ **Delivery Confirmation** - Tracking and retry mechanisms
✅ **Dashboard Integration** - Complete UI management

The system is production-ready and will automatically handle all email notifications for the Smart Waste Management System.