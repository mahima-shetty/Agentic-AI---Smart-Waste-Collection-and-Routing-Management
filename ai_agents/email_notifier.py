"""
Email Notification System for Smart Waste Management
Handles SMTP integration, email templates, and delivery confirmation
"""

import smtplib
import ssl
import logging
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import sqlite3
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class EmailConfig:
    """Email configuration settings"""
    smtp_server: str
    smtp_port: int
    use_tls: bool
    sender_email: str
    sender_password: str
    sender_name: str = "Smart Waste Management System"

@dataclass
class EmailTemplate:
    """Email template structure"""
    template_id: str
    subject_template: str
    body_template: str
    is_html: bool = True
    priority: str = "normal"  # "low", "normal", "high", "urgent"

@dataclass
class EmailNotification:
    """Email notification data"""
    notification_id: str
    recipient_email: str
    recipient_name: str
    template_id: str
    template_data: Dict[str, Any]
    priority: str = "normal"
    created_at: datetime = None
    sent_at: Optional[datetime] = None
    delivery_status: str = "pending"  # "pending", "sent", "failed", "bounced"
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

class EmailNotificationSystem:
    """
    Email notification system for waste management alerts
    Handles SMTP integration, templates, and delivery tracking
    """
    
    def __init__(self, config_path: str = "data/config.json", db_path: str = "backend/db/operators.db"):
        self.config_path = config_path
        self.db_path = db_path
        self.email_config: Optional[EmailConfig] = None
        self.templates: Dict[str, EmailTemplate] = {}
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Email delivery statistics
        self.stats = {
            "total_sent": 0,
            "total_failed": 0,
            "total_pending": 0,
            "delivery_rate": 0.0,
            "last_updated": datetime.now()
        }
        
        # Initialize system
        self._load_configuration()
        self._setup_database()
        self._initialize_templates()
        
        logger.info("📧 Email Notification System initialized")
    
    def _load_configuration(self):
        """Load email configuration from config file"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            email_config = config_data.get("email_notifications", {})
            
            self.email_config = EmailConfig(
                smtp_server=email_config.get("smtp_server", "smtp.gmail.com"),
                smtp_port=email_config.get("smtp_port", 587),
                use_tls=email_config.get("use_tls", True),
                sender_email=email_config.get("sender_email", ""),
                sender_password=email_config.get("sender_password", ""),
                sender_name=email_config.get("sender_name", "Smart Waste Management System")
            )
            
            if not self.email_config.sender_email or not self.email_config.sender_password:
                logger.warning("⚠️ Email credentials not configured. Email notifications will be disabled.")
                self.email_config = None
            else:
                logger.info("✅ Email configuration loaded successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to load email configuration: {e}")
            self.email_config = None
    
    def _setup_database(self):
        """Setup database tables for email notifications"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create email_notifications table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_notifications (
                    notification_id TEXT PRIMARY KEY,
                    recipient_email TEXT NOT NULL,
                    recipient_name TEXT NOT NULL,
                    template_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    body TEXT NOT NULL,
                    priority TEXT DEFAULT 'normal',
                    created_at TEXT NOT NULL,
                    sent_at TEXT,
                    delivery_status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3
                )
            """)
            
            # Create email_preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_preferences (
                    operator_id INTEGER PRIMARY KEY,
                    email TEXT NOT NULL,
                    alert_notifications BOOLEAN DEFAULT 1,
                    escalation_notifications BOOLEAN DEFAULT 1,
                    daily_reports BOOLEAN DEFAULT 1,
                    weekly_summaries BOOLEAN DEFAULT 1,
                    emergency_alerts BOOLEAN DEFAULT 1,
                    notification_frequency TEXT DEFAULT 'immediate',
                    quiet_hours_start TEXT DEFAULT '22:00',
                    quiet_hours_end TEXT DEFAULT '06:00',
                    FOREIGN KEY (operator_id) REFERENCES operators (id)
                )
            """)
            
            # Create supervisor_contacts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS supervisor_contacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ward_id INTEGER NOT NULL,
                    supervisor_name TEXT NOT NULL,
                    supervisor_email TEXT NOT NULL,
                    escalation_level INTEGER DEFAULT 1,
                    contact_priority INTEGER DEFAULT 1,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("📊 Email notification database tables created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup email database: {e}")
            raise
    
    def _initialize_templates(self):
        """Initialize email templates for different alert types"""
        try:
            # Overflow Warning Template
            self.templates["overflow_warning"] = EmailTemplate(
                template_id="overflow_warning",
                subject_template="⚠️ Bin Overflow Warning - {bin_id} ({ward_name})",
                body_template="""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <div style="background: linear-gradient(135deg, #ff9800, #f57c00); color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                            <h2 style="margin: 0; font-size: 24px;">⚠️ Bin Overflow Warning</h2>
                            <p style="margin: 5px 0 0 0; opacity: 0.9;">Smart Waste Management Alert</p>
                        </div>
                        
                        <div style="background: #f8f9fa; padding: 20px; border: 1px solid #dee2e6;">
                            <h3 style="color: #ff9800; margin-top: 0;">Alert Details</h3>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold; width: 30%;">Bin ID:</td>
                                    <td style="padding: 8px 0;">{bin_id}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold;">Ward:</td>
                                    <td style="padding: 8px 0;">{ward_name}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold;">Fill Level:</td>
                                    <td style="padding: 8px 0; color: #ff9800; font-weight: bold;">{fill_level}%</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold;">Location:</td>
                                    <td style="padding: 8px 0;">{location}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold;">Alert Time:</td>
                                    <td style="padding: 8px 0;">{alert_time}</td>
                                </tr>
                            </table>
                        </div>
                        
                        <div style="background: white; padding: 20px; border: 1px solid #dee2e6; border-top: none;">
                            <h3 style="color: #333; margin-top: 0;">Recommended Action</h3>
                            <p style="background: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ff9800;">
                                {recommended_action}
                            </p>
                            
                            <h3 style="color: #333;">Additional Information</h3>
                            <p>{natural_language_summary}</p>
                            
                            <div style="margin-top: 20px; padding: 15px; background: #e3f2fd; border-radius: 5px;">
                                <p style="margin: 0; font-size: 14px; color: #1976d2;">
                                    <strong>Priority Level:</strong> {priority} | 
                                    <strong>Bin Type:</strong> {bin_type} | 
                                    <strong>Fill Rate:</strong> {fill_rate} L/h
                                </p>
                            </div>
                        </div>
                        
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 0 0 8px 8px; text-align: center; font-size: 12px; color: #666;">
                            <p style="margin: 0;">Smart Waste Management System | BMC Mumbai</p>
                            <p style="margin: 5px 0 0 0;">Generated at {timestamp}</p>
                        </div>
                    </div>
                </body>
                </html>
                """,
                is_html=True,
                priority="high"
            )
            
            # Critical Overflow Template
            self.templates["overflow_critical"] = EmailTemplate(
                template_id="overflow_critical",
                subject_template="🚨 CRITICAL: Bin Overflow - Immediate Action Required - {bin_id}",
                body_template="""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <div style="background: linear-gradient(135deg, #d32f2f, #b71c1c); color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                            <h2 style="margin: 0; font-size: 26px;">🚨 CRITICAL OVERFLOW ALERT</h2>
                            <p style="margin: 5px 0 0 0; opacity: 0.9; font-weight: bold;">IMMEDIATE ACTION REQUIRED</p>
                        </div>
                        
                        <div style="background: #ffebee; padding: 20px; border: 2px solid #f44336;">
                            <h3 style="color: #d32f2f; margin-top: 0;">🚨 URGENT ALERT DETAILS</h3>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold; width: 30%;">Bin ID:</td>
                                    <td style="padding: 8px 0; font-weight: bold; color: #d32f2f;">{bin_id}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold;">Ward:</td>
                                    <td style="padding: 8px 0;">{ward_name}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold;">Fill Level:</td>
                                    <td style="padding: 8px 0; color: #d32f2f; font-weight: bold; font-size: 18px;">{fill_level}%</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold;">Location:</td>
                                    <td style="padding: 8px 0;">{location}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold;">Alert Time:</td>
                                    <td style="padding: 8px 0;">{alert_time}</td>
                                </tr>
                            </table>
                        </div>
                        
                        <div style="background: white; padding: 20px; border: 2px solid #f44336; border-top: none;">
                            <h3 style="color: #d32f2f; margin-top: 0;">⚡ IMMEDIATE ACTION REQUIRED</h3>
                            <div style="background: #ffcdd2; padding: 15px; border-radius: 5px; border-left: 4px solid #d32f2f;">
                                <p style="margin: 0; font-weight: bold; color: #d32f2f;">{recommended_action}</p>
                            </div>
                            
                            <h3 style="color: #333;">Situation Analysis</h3>
                            <p>{natural_language_summary}</p>
                            
                            <div style="margin-top: 20px; padding: 15px; background: #fff3e0; border-radius: 5px; border: 1px solid #ff9800;">
                                <p style="margin: 0; font-size: 14px; color: #e65100;">
                                    <strong>⚠️ This is a critical alert requiring immediate response</strong><br>
                                    <strong>Priority:</strong> URGENT | 
                                    <strong>Bin Type:</strong> {bin_type} | 
                                    <strong>Fill Rate:</strong> {fill_rate} L/h
                                </p>
                            </div>
                        </div>
                        
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 0 0 8px 8px; text-align: center; font-size: 12px; color: #666;">
                            <p style="margin: 0;">Smart Waste Management System | BMC Mumbai</p>
                            <p style="margin: 5px 0 0 0;">Generated at {timestamp}</p>
                        </div>
                    </div>
                </body>
                </html>
                """,
                is_html=True,
                priority="urgent"
            )
            
            # Predictive Alert Template
            self.templates["overflow_prediction"] = EmailTemplate(
                template_id="overflow_prediction",
                subject_template="🔮 Predictive Alert: Bin {bin_id} - Overflow Expected in {hours_until}h",
                body_template="""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <div style="background: linear-gradient(135deg, #2196f3, #1976d2); color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                            <h2 style="margin: 0; font-size: 24px;">🔮 Predictive Overflow Alert</h2>
                            <p style="margin: 5px 0 0 0; opacity: 0.9;">AI-Powered Prediction</p>
                        </div>
                        
                        <div style="background: #e3f2fd; padding: 20px; border: 1px solid #2196f3;">
                            <h3 style="color: #1976d2; margin-top: 0;">Prediction Details</h3>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold; width: 30%;">Bin ID:</td>
                                    <td style="padding: 8px 0;">{bin_id}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold;">Ward:</td>
                                    <td style="padding: 8px 0;">{ward_name}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold;">Current Fill:</td>
                                    <td style="padding: 8px 0; color: #1976d2; font-weight: bold;">{fill_level}%</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold;">Predicted Overflow:</td>
                                    <td style="padding: 8px 0; color: #f57c00; font-weight: bold;">{predicted_time}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold;">Time Remaining:</td>
                                    <td style="padding: 8px 0; color: #f57c00; font-weight: bold;">{hours_until} hours</td>
                                </tr>
                            </table>
                        </div>
                        
                        <div style="background: white; padding: 20px; border: 1px solid #2196f3; border-top: none;">
                            <h3 style="color: #333; margin-top: 0;">Recommended Proactive Action</h3>
                            <p style="background: #e1f5fe; padding: 15px; border-radius: 5px; border-left: 4px solid #2196f3;">
                                {recommended_action}
                            </p>
                            
                            <h3 style="color: #333;">AI Analysis</h3>
                            <p>{natural_language_summary}</p>
                            
                            <div style="margin-top: 20px; padding: 15px; background: #f3e5f5; border-radius: 5px;">
                                <p style="margin: 0; font-size: 14px; color: #7b1fa2;">
                                    <strong>🤖 AI Confidence:</strong> High | 
                                    <strong>Bin Type:</strong> {bin_type} | 
                                    <strong>Fill Rate:</strong> {fill_rate} L/h
                                </p>
                            </div>
                        </div>
                        
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 0 0 8px 8px; text-align: center; font-size: 12px; color: #666;">
                            <p style="margin: 0;">Smart Waste Management System | BMC Mumbai</p>
                            <p style="margin: 5px 0 0 0;">Generated at {timestamp}</p>
                        </div>
                    </div>
                </body>
                </html>
                """,
                is_html=True,
                priority="normal"
            )
            
            # Escalation Template
            self.templates["escalation"] = EmailTemplate(
                template_id="escalation",
                subject_template="🚨 ESCALATION: Multiple Critical Alerts - Ward {ward_name}",
                body_template="""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <div style="background: linear-gradient(135deg, #e91e63, #c2185b); color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                            <h2 style="margin: 0; font-size: 26px;">🚨 SUPERVISOR ESCALATION</h2>
                            <p style="margin: 5px 0 0 0; opacity: 0.9; font-weight: bold;">MULTIPLE CRITICAL ALERTS</p>
                        </div>
                        
                        <div style="background: #fce4ec; padding: 20px; border: 2px solid #e91e63;">
                            <h3 style="color: #c2185b; margin-top: 0;">Escalation Summary</h3>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold; width: 30%;">Ward:</td>
                                    <td style="padding: 8px 0; font-weight: bold; color: #c2185b;">{ward_name}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold;">Critical Alerts:</td>
                                    <td style="padding: 8px 0; color: #c2185b; font-weight: bold; font-size: 18px;">{critical_count}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold;">Escalation Time:</td>
                                    <td style="padding: 8px 0;">{escalation_time}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; font-weight: bold;">Escalation Reason:</td>
                                    <td style="padding: 8px 0;">{escalation_reason}</td>
                                </tr>
                            </table>
                        </div>
                        
                        <div style="background: white; padding: 20px; border: 2px solid #e91e63; border-top: none;">
                            <h3 style="color: #c2185b; margin-top: 0;">Critical Bins Requiring Attention</h3>
                            <div style="background: #ffebee; padding: 15px; border-radius: 5px;">
                                {critical_bins_list}
                            </div>
                            
                            <h3 style="color: #333;">Recommended Supervisor Action</h3>
                            <div style="background: #f3e5f5; padding: 15px; border-radius: 5px; border-left: 4px solid #e91e63;">
                                <p style="margin: 0; font-weight: bold; color: #c2185b;">
                                    • Deploy additional collection vehicles immediately<br>
                                    • Coordinate emergency collection routes<br>
                                    • Contact field supervisors for immediate response<br>
                                    • Monitor situation closely for next 2 hours
                                </p>
                            </div>
                        </div>
                        
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 0 0 8px 8px; text-align: center; font-size: 12px; color: #666;">
                            <p style="margin: 0;">Smart Waste Management System | BMC Mumbai</p>
                            <p style="margin: 5px 0 0 0;">Generated at {timestamp}</p>
                        </div>
                    </div>
                </body>
                </html>
                """,
                is_html=True,
                priority="urgent"
            )
            
            logger.info("📧 Email templates initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize email templates: {e}")
            raise
    
    async def send_alert_notification(
        self,
        alert_data: Dict[str, Any],
        bin_info: Dict[str, Any],
        recipient_email: str,
        recipient_name: str
    ) -> bool:
        """Send alert notification email"""
        try:
            if not self.email_config:
                logger.warning("⚠️ Email not configured, skipping notification")
                return False
            
            alert_type = alert_data.get("alert_type", "overflow_warning")
            template = self.templates.get(alert_type)
            
            if not template:
                logger.error(f"❌ Template not found for alert type: {alert_type}")
                return False
            
            # Prepare template data
            template_data = {
                "bin_id": bin_info.get("id", "Unknown"),
                "ward_name": f"Ward {bin_info.get('ward_id', 'Unknown')}",
                "fill_level": bin_info.get("current_fill", 0),
                "location": f"Lat: {bin_info.get('latitude', 'N/A')}, Lon: {bin_info.get('longitude', 'N/A')}",
                "alert_time": alert_data.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "recommended_action": alert_data.get("recommended_action", "Please check bin status"),
                "natural_language_summary": alert_data.get("natural_language_summary", "Alert generated for bin monitoring"),
                "priority": alert_data.get("priority", "Normal"),
                "bin_type": bin_info.get("bin_type", "residential"),
                "fill_rate": bin_info.get("fill_rate", 0),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add specific data for predictive alerts
            if alert_type == "overflow_prediction":
                predicted_time = alert_data.get("predicted_overflow_time")
                if predicted_time:
                    if isinstance(predicted_time, str):
                        predicted_time = datetime.fromisoformat(predicted_time)
                    template_data["predicted_time"] = predicted_time.strftime("%Y-%m-%d %H:%M:%S")
                    hours_until = (predicted_time - datetime.now()).total_seconds() / 3600
                    template_data["hours_until"] = f"{hours_until:.1f}"
                else:
                    template_data["predicted_time"] = "Unknown"
                    template_data["hours_until"] = "Unknown"
            
            # Create notification
            notification = EmailNotification(
                notification_id=f"email_{alert_data.get('id', 'unknown')}_{datetime.now().timestamp()}",
                recipient_email=recipient_email,
                recipient_name=recipient_name,
                template_id=template.template_id,
                template_data=template_data,
                priority=template.priority,
                created_at=datetime.now()
            )
            
            # Send email
            success = await self._send_email_notification(notification, template)
            
            if success:
                logger.info(f"📧 Alert notification sent to {recipient_email}")
            else:
                logger.error(f"❌ Failed to send alert notification to {recipient_email}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to send alert notification: {e}")
            return False
    
    async def send_escalation_notification(
        self,
        escalation_data: Dict[str, Any],
        supervisor_contacts: List[Dict[str, Any]]
    ) -> bool:
        """Send escalation notification to supervisors"""
        try:
            if not self.email_config:
                logger.warning("⚠️ Email not configured, skipping escalation notification")
                return False
            
            template = self.templates.get("escalation")
            if not template:
                logger.error("❌ Escalation template not found")
                return False
            
            # Prepare critical bins list
            critical_alerts = escalation_data.get("critical_alerts", [])
            critical_bins_html = ""
            for alert in critical_alerts:
                critical_bins_html += f"""
                <div style="margin: 10px 0; padding: 10px; background: white; border-left: 4px solid #d32f2f;">
                    <strong>Bin {alert.get('bin_id', 'Unknown')}</strong> - 
                    Fill Level: <span style="color: #d32f2f; font-weight: bold;">{alert.get('fill_level', 'N/A')}%</span><br>
                    <small>Alert: {alert.get('message', 'Critical overflow alert')}</small>
                </div>
                """
            
            # Prepare template data
            template_data = {
                "ward_name": escalation_data.get("ward_name", "Unknown"),
                "critical_count": len(critical_alerts),
                "escalation_time": escalation_data.get("escalation_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "escalation_reason": escalation_data.get("escalation_reason", "Multiple critical alerts require immediate attention"),
                "critical_bins_list": critical_bins_html,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Send to all supervisors
            success_count = 0
            for supervisor in supervisor_contacts:
                notification = EmailNotification(
                    notification_id=f"escalation_{escalation_data.get('escalation_id', 'unknown')}_{supervisor.get('supervisor_email', 'unknown')}_{datetime.now().timestamp()}",
                    recipient_email=supervisor.get("supervisor_email", ""),
                    recipient_name=supervisor.get("supervisor_name", "Supervisor"),
                    template_id=template.template_id,
                    template_data=template_data,
                    priority=template.priority,
                    created_at=datetime.now()
                )
                
                if await self._send_email_notification(notification, template):
                    success_count += 1
            
            logger.info(f"📧 Escalation notifications sent to {success_count}/{len(supervisor_contacts)} supervisors")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to send escalation notifications: {e}")
            return False
    
    async def _send_email_notification(self, notification: EmailNotification, template: EmailTemplate) -> bool:
        """Send individual email notification"""
        try:
            # Format subject and body
            subject = template.subject_template.format(**notification.template_data)
            body = template.body_template.format(**notification.template_data)
            
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['From'] = f"{self.email_config.sender_name} <{self.email_config.sender_email}>"
            msg['To'] = notification.recipient_email
            msg['Subject'] = subject
            
            # Set priority headers
            if template.priority == "urgent":
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'
                msg['Importance'] = 'High'
            elif template.priority == "high":
                msg['X-Priority'] = '2'
                msg['X-MSMail-Priority'] = 'High'
                msg['Importance'] = 'High'
            
            # Add body
            if template.is_html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Send email in thread pool to avoid blocking
            success = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._send_smtp_email,
                msg
            )
            
            # Update notification status
            if success:
                notification.delivery_status = "sent"
                notification.sent_at = datetime.now()
                self.stats["total_sent"] += 1
            else:
                notification.delivery_status = "failed"
                notification.retry_count += 1
                self.stats["total_failed"] += 1
            
            # Save to database
            await self._save_notification_to_database(notification, subject, body)
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to send email notification: {e}")
            notification.delivery_status = "failed"
            notification.error_message = str(e)
            notification.retry_count += 1
            self.stats["total_failed"] += 1
            return False
    
    def _send_smtp_email(self, msg: MIMEMultipart) -> bool:
        """Send email via SMTP (runs in thread pool)"""
        try:
            # Create SMTP connection
            if self.email_config.use_tls:
                context = ssl.create_default_context()
                server = smtplib.SMTP(self.email_config.smtp_server, self.email_config.smtp_port)
                server.starttls(context=context)
            else:
                server = smtplib.SMTP(self.email_config.smtp_server, self.email_config.smtp_port)
            
            # Login and send
            server.login(self.email_config.sender_email, self.email_config.sender_password)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ SMTP send failed: {e}")
            return False
    
    async def _save_notification_to_database(self, notification: EmailNotification, subject: str, body: str):
        """Save notification to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO email_notifications (
                    notification_id, recipient_email, recipient_name, template_id,
                    subject, body, priority, created_at, sent_at, delivery_status,
                    error_message, retry_count, max_retries
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                notification.notification_id,
                notification.recipient_email,
                notification.recipient_name,
                notification.template_id,
                subject,
                body,
                notification.priority,
                notification.created_at.isoformat(),
                notification.sent_at.isoformat() if notification.sent_at else None,
                notification.delivery_status,
                notification.error_message,
                notification.retry_count,
                notification.max_retries
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to save notification to database: {e}")
    
    async def get_operator_email_preferences(self, operator_id: int) -> Dict[str, Any]:
        """Get email preferences for an operator"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM email_preferences WHERE operator_id = ?
            """, (operator_id,))
            
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                preferences = dict(zip(columns, row))
            else:
                # Create default preferences
                preferences = {
                    "operator_id": operator_id,
                    "alert_notifications": True,
                    "escalation_notifications": True,
                    "daily_reports": True,
                    "weekly_summaries": True,
                    "emergency_alerts": True,
                    "notification_frequency": "immediate",
                    "quiet_hours_start": "22:00",
                    "quiet_hours_end": "06:00"
                }
                
                # Save default preferences
                cursor.execute("""
                    INSERT OR IGNORE INTO email_preferences (
                        operator_id, email, alert_notifications, escalation_notifications,
                        daily_reports, weekly_summaries, emergency_alerts,
                        notification_frequency, quiet_hours_start, quiet_hours_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    operator_id, "", True, True, True, True, True,
                    "immediate", "22:00", "06:00"
                ))
                conn.commit()
            
            conn.close()
            return preferences
            
        except Exception as e:
            logger.error(f"❌ Failed to get email preferences: {e}")
            return {}
    
    async def get_supervisor_contacts(self, ward_id: int) -> List[Dict[str, Any]]:
        """Get supervisor contacts for a ward"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM supervisor_contacts 
                WHERE ward_id = ? AND is_active = 1
                ORDER BY escalation_level, contact_priority
            """, (ward_id,))
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            supervisors = []
            for row in rows:
                supervisor = dict(zip(columns, row))
                supervisors.append(supervisor)
            
            # If no supervisors found, create default entries
            if not supervisors:
                default_supervisors = [
                    {
                        "ward_id": ward_id,
                        "supervisor_name": f"Ward {ward_id} Supervisor",
                        "supervisor_email": f"supervisor.ward{ward_id}@bmc.gov.in",
                        "escalation_level": 1,
                        "contact_priority": 1,
                        "is_active": True
                    }
                ]
                
                for supervisor in default_supervisors:
                    cursor.execute("""
                        INSERT OR IGNORE INTO supervisor_contacts (
                            ward_id, supervisor_name, supervisor_email,
                            escalation_level, contact_priority, is_active
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        supervisor["ward_id"],
                        supervisor["supervisor_name"],
                        supervisor["supervisor_email"],
                        supervisor["escalation_level"],
                        supervisor["contact_priority"],
                        supervisor["is_active"]
                    ))
                
                conn.commit()
                supervisors = default_supervisors
            
            conn.close()
            return supervisors
            
        except Exception as e:
            logger.error(f"❌ Failed to get supervisor contacts: {e}")
            return []
    
    def get_email_stats(self) -> Dict[str, Any]:
        """Get email notification statistics"""
        try:
            # Update delivery rate
            total_emails = self.stats["total_sent"] + self.stats["total_failed"]
            if total_emails > 0:
                self.stats["delivery_rate"] = (self.stats["total_sent"] / total_emails) * 100
            
            self.stats["last_updated"] = datetime.now().isoformat()
            return self.stats.copy()
            
        except Exception as e:
            logger.error(f"❌ Failed to get email stats: {e}")
            return {}
    
    async def retry_failed_notifications(self) -> int:
        """Retry failed email notifications"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM email_notifications 
                WHERE delivery_status = 'failed' AND retry_count < max_retries
                ORDER BY created_at DESC
                LIMIT 10
            """)
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            retry_count = 0
            for row in rows:
                notification_data = dict(zip(columns, row))
                
                # Recreate notification object
                notification = EmailNotification(
                    notification_id=notification_data["notification_id"],
                    recipient_email=notification_data["recipient_email"],
                    recipient_name=notification_data["recipient_name"],
                    template_id=notification_data["template_id"],
                    template_data={},  # Would need to reconstruct from subject/body
                    priority=notification_data["priority"],
                    created_at=datetime.fromisoformat(notification_data["created_at"]),
                    delivery_status=notification_data["delivery_status"],
                    retry_count=notification_data["retry_count"],
                    max_retries=notification_data["max_retries"]
                )
                
                # Get template
                template = self.templates.get(notification.template_id)
                if template:
                    # Create simple message for retry
                    msg = MIMEMultipart()
                    msg['From'] = f"{self.email_config.sender_name} <{self.email_config.sender_email}>"
                    msg['To'] = notification.recipient_email
                    msg['Subject'] = notification_data["subject"]
                    msg.attach(MIMEText(notification_data["body"], 'html' if template.is_html else 'plain'))
                    
                    # Retry sending
                    success = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self._send_smtp_email,
                        msg
                    )
                    
                    if success:
                        notification.delivery_status = "sent"
                        notification.sent_at = datetime.now()
                        retry_count += 1
                    else:
                        notification.retry_count += 1
                    
                    # Update database
                    await self._save_notification_to_database(notification, notification_data["subject"], notification_data["body"])
            
            conn.close()
            
            if retry_count > 0:
                logger.info(f"📧 Successfully retried {retry_count} failed notifications")
            
            return retry_count
            
        except Exception as e:
            logger.error(f"❌ Failed to retry notifications: {e}")
            return 0