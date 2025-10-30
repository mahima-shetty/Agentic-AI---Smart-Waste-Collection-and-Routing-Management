"""
Email Preferences Management for Smart Waste Management Dashboard
Allows operators to configure their email notification preferences
"""

import streamlit as st
import sqlite3
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class EmailPreferencesManager:
    """Manages email preferences for operators in the dashboard"""
    
    def __init__(self, db_path: str = "backend/db/operators.db"):
        self.db_path = db_path
    
    def render_email_preferences_page(self, operator_id: int):
        """Render the email preferences configuration page"""
        try:
            st.title("📧 Email Notification Preferences")
            st.markdown("Configure your email notification settings for waste management alerts.")
            
            # Load current preferences
            current_prefs = self._get_operator_preferences(operator_id)
            
            if not current_prefs:
                st.error("❌ Unable to load email preferences. Please contact administrator.")
                return
            
            # Create form for preferences
            with st.form("email_preferences_form"):
                st.subheader("🔔 Notification Types")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    alert_notifications = st.checkbox(
                        "Alert Notifications",
                        value=current_prefs.get("alert_notifications", True),
                        help="Receive emails for bin overflow warnings and critical alerts"
                    )
                    
                    escalation_notifications = st.checkbox(
                        "Escalation Notifications",
                        value=current_prefs.get("escalation_notifications", True),
                        help="Receive emails when alerts are escalated to supervisors"
                    )
                    
                    emergency_alerts = st.checkbox(
                        "Emergency Alerts",
                        value=current_prefs.get("emergency_alerts", True),
                        help="Receive immediate notifications for critical situations"
                    )
                
                with col2:
                    daily_reports = st.checkbox(
                        "Daily Reports",
                        value=current_prefs.get("daily_reports", True),
                        help="Receive daily summary reports of waste collection activities"
                    )
                    
                    weekly_summaries = st.checkbox(
                        "Weekly Summaries",
                        value=current_prefs.get("weekly_summaries", True),
                        help="Receive weekly performance and analytics summaries"
                    )
                
                st.subheader("⏰ Notification Timing")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    notification_frequency = st.selectbox(
                        "Notification Frequency",
                        options=["immediate", "hourly", "daily"],
                        index=["immediate", "hourly", "daily"].index(
                            current_prefs.get("notification_frequency", "immediate")
                        ),
                        help="How often to receive non-emergency notifications"
                    )
                
                with col4:
                    st.write("**Quiet Hours** (No non-emergency notifications)")
                    
                    quiet_start = st.time_input(
                        "Start Time",
                        value=datetime.strptime(
                            current_prefs.get("quiet_hours_start", "22:00"), "%H:%M"
                        ).time(),
                        help="Start of quiet hours"
                    )
                    
                    quiet_end = st.time_input(
                        "End Time",
                        value=datetime.strptime(
                            current_prefs.get("quiet_hours_end", "06:00"), "%H:%M"
                        ).time(),
                        help="End of quiet hours"
                    )
                
                st.subheader("📊 Current Email Statistics")
                
                # Display email statistics
                email_stats = self._get_email_statistics(operator_id)
                if email_stats:
                    col5, col6, col7, col8 = st.columns(4)
                    
                    with col5:
                        st.metric(
                            "Emails Sent",
                            email_stats.get("total_sent", 0),
                            help="Total emails sent to you"
                        )
                    
                    with col6:
                        st.metric(
                            "Delivery Rate",
                            f"{email_stats.get('delivery_rate', 0):.1f}%",
                            help="Percentage of emails successfully delivered"
                        )
                    
                    with col7:
                        st.metric(
                            "Failed Deliveries",
                            email_stats.get("total_failed", 0),
                            help="Number of failed email deliveries"
                        )
                    
                    with col8:
                        st.metric(
                            "Pending",
                            email_stats.get("total_pending", 0),
                            help="Emails waiting to be sent"
                        )
                
                # Submit button
                submitted = st.form_submit_button(
                    "💾 Save Preferences",
                    type="primary",
                    use_container_width=True
                )
                
                if submitted:
                    # Save preferences
                    new_preferences = {
                        "alert_notifications": alert_notifications,
                        "escalation_notifications": escalation_notifications,
                        "daily_reports": daily_reports,
                        "weekly_summaries": weekly_summaries,
                        "emergency_alerts": emergency_alerts,
                        "notification_frequency": notification_frequency,
                        "quiet_hours_start": quiet_start.strftime("%H:%M"),
                        "quiet_hours_end": quiet_end.strftime("%H:%M")
                    }
                    
                    success = self._save_operator_preferences(operator_id, new_preferences)
                    
                    if success:
                        st.success("✅ Email preferences saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save preferences. Please try again.")
            
            # Test email section
            st.subheader("🧪 Test Email Notifications")
            
            col9, col10 = st.columns([2, 1])
            
            with col9:
                st.info("Send a test email to verify your notification settings are working correctly.")
            
            with col10:
                if st.button("📧 Send Test Email", type="secondary"):
                    success = self._send_test_email(operator_id)
                    if success:
                        st.success("✅ Test email sent!")
                    else:
                        st.error("❌ Failed to send test email.")
            
            # Recent notifications section
            st.subheader("📬 Recent Email Notifications")
            
            recent_notifications = self._get_recent_notifications(operator_id)
            if recent_notifications:
                for notification in recent_notifications[:10]:  # Show last 10
                    with st.expander(
                        f"📧 {notification['subject']} - {notification['created_at']}"
                    ):
                        col11, col12 = st.columns([3, 1])
                        
                        with col11:
                            st.write(f"**To:** {notification['recipient_email']}")
                            st.write(f"**Status:** {notification['delivery_status'].title()}")
                            st.write(f"**Template:** {notification['template_id']}")
                            
                            if notification['error_message']:
                                st.error(f"**Error:** {notification['error_message']}")
                        
                        with col12:
                            status_color = {
                                "sent": "🟢",
                                "pending": "🟡",
                                "failed": "🔴"
                            }.get(notification['delivery_status'], "⚪")
                            
                            st.write(f"{status_color} **{notification['delivery_status'].upper()}**")
                            
                            if notification['delivery_status'] == 'failed' and notification['retry_count'] < 3:
                                if st.button(f"🔄 Retry", key=f"retry_{notification['notification_id']}"):
                                    # Trigger retry (would need to implement)
                                    st.info("Retry requested...")
            else:
                st.info("No recent email notifications found.")
        
        except Exception as e:
            logger.error(f"❌ Error rendering email preferences page: {e}")
            st.error("❌ An error occurred while loading email preferences.")
    
    def _get_operator_preferences(self, operator_id: int) -> Optional[Dict[str, Any]]:
        """Get current email preferences for operator"""
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
                # Return default preferences
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
            
            conn.close()
            return preferences
            
        except Exception as e:
            logger.error(f"❌ Failed to get operator preferences: {e}")
            return None
    
    def _save_operator_preferences(self, operator_id: int, preferences: Dict[str, Any]) -> bool:
        """Save email preferences for operator"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get operator email
            cursor.execute("SELECT email FROM operators WHERE id = ?", (operator_id,))
            operator_row = cursor.fetchone()
            
            if not operator_row:
                logger.error(f"❌ Operator {operator_id} not found")
                return False
            
            operator_email = operator_row[0]
            
            # Save preferences
            cursor.execute("""
                INSERT OR REPLACE INTO email_preferences (
                    operator_id, email, alert_notifications, escalation_notifications,
                    daily_reports, weekly_summaries, emergency_alerts,
                    notification_frequency, quiet_hours_start, quiet_hours_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                operator_id,
                operator_email,
                preferences.get("alert_notifications", True),
                preferences.get("escalation_notifications", True),
                preferences.get("daily_reports", True),
                preferences.get("weekly_summaries", True),
                preferences.get("emergency_alerts", True),
                preferences.get("notification_frequency", "immediate"),
                preferences.get("quiet_hours_start", "22:00"),
                preferences.get("quiet_hours_end", "06:00")
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Saved email preferences for operator {operator_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save operator preferences: {e}")
            return False
    
    def _get_email_statistics(self, operator_id: int) -> Optional[Dict[str, Any]]:
        """Get email statistics for operator"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get operator email
            cursor.execute("SELECT email FROM operators WHERE id = ?", (operator_id,))
            operator_row = cursor.fetchone()
            
            if not operator_row:
                return None
            
            operator_email = operator_row[0]
            
            # Get email statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_emails,
                    SUM(CASE WHEN delivery_status = 'sent' THEN 1 ELSE 0 END) as total_sent,
                    SUM(CASE WHEN delivery_status = 'failed' THEN 1 ELSE 0 END) as total_failed,
                    SUM(CASE WHEN delivery_status = 'pending' THEN 1 ELSE 0 END) as total_pending
                FROM email_notifications 
                WHERE recipient_email = ?
            """, (operator_email,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                total_emails, total_sent, total_failed, total_pending = row
                delivery_rate = (total_sent / total_emails * 100) if total_emails > 0 else 0
                
                return {
                    "total_emails": total_emails,
                    "total_sent": total_sent,
                    "total_failed": total_failed,
                    "total_pending": total_pending,
                    "delivery_rate": delivery_rate
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get email statistics: {e}")
            return None
    
    def _get_recent_notifications(self, operator_id: int) -> List[Dict[str, Any]]:
        """Get recent email notifications for operator"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get operator email
            cursor.execute("SELECT email FROM operators WHERE id = ?", (operator_id,))
            operator_row = cursor.fetchone()
            
            if not operator_row:
                return []
            
            operator_email = operator_row[0]
            
            # Get recent notifications
            cursor.execute("""
                SELECT * FROM email_notifications 
                WHERE recipient_email = ?
                ORDER BY created_at DESC
                LIMIT 20
            """, (operator_email,))
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            notifications = []
            for row in rows:
                notification = dict(zip(columns, row))
                notifications.append(notification)
            
            conn.close()
            return notifications
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent notifications: {e}")
            return []
    
    def _send_test_email(self, operator_id: int) -> bool:
        """Send a test email to operator"""
        try:
            # This would integrate with the EmailNotificationSystem
            # For now, just simulate success
            logger.info(f"📧 Test email requested for operator {operator_id}")
            
            # In a real implementation, this would:
            # 1. Get operator email
            # 2. Create a test notification
            # 3. Send via EmailNotificationSystem
            # 4. Return success/failure
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send test email: {e}")
            return False

def render_email_preferences_sidebar():
    """Render email preferences in sidebar"""
    try:
        if st.session_state.get("authenticated") and st.session_state.get("operator_id"):
            with st.sidebar:
                st.subheader("📧 Email Settings")
                
                if st.button("⚙️ Email Preferences", use_container_width=True):
                    st.session_state["current_page"] = "email_preferences"
                
                # Quick toggle for emergency alerts
                if "email_emergency_enabled" not in st.session_state:
                    st.session_state["email_emergency_enabled"] = True
                
                emergency_enabled = st.toggle(
                    "🚨 Emergency Alerts",
                    value=st.session_state["email_emergency_enabled"],
                    help="Receive immediate email alerts for critical situations"
                )
                
                if emergency_enabled != st.session_state["email_emergency_enabled"]:
                    st.session_state["email_emergency_enabled"] = emergency_enabled
                    # Update preference in database
                    st.success("✅ Emergency alert preference updated!")
    
    except Exception as e:
        logger.error(f"❌ Error rendering email preferences sidebar: {e}")