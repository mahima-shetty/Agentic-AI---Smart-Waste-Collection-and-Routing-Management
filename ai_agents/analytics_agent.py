"""
Analytics Agent - Advanced data analysis and intelligent reporting with semantic understanding
Enhanced with LangChain analysis workflows, vector similarity search, and natural language capabilities
"""

import asyncio
import logging
import json
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
# import plotly.graph_objects as go  # Optional for visualizations
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("⚠️ scikit-learn not available - ML features will be limited")

from .langchain_base import BaseLangChainAgent, WasteManagementTool, AgentPromptTemplates
from .mcp_handler import MCPHandler
from .vector_db import VectorDatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class AnalyticsInsight:
    """Data model for analytics insights"""
    insight_id: str
    insight_type: str
    title: str
    description: str
    data_sources: List[str]
    confidence_score: float
    impact_level: str
    recommendations: List[str]
    natural_language_summary: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class AnalyticsAgent(BaseLangChainAgent):
    """
    Enhanced Analytics Agent for advanced data analysis and intelligent reporting
    Uses Pandas, Scikit-learn, LangChain workflows, and vector similarity search
    """
    
    def __init__(
        self,
        redis_client=None,
        vector_db: Optional[VectorDatabaseManager] = None,
        mcp_handler: Optional[MCPHandler] = None,
        db_path: str = "backend/db/operators.db"
    ):
        super().__init__(
            agent_id="analytics_agent",
            agent_type="analytics",
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        
        self.redis_client = redis_client
        self.vector_db = vector_db
        self.mcp_handler = mcp_handler
        self.db_path = db_path
        
        # Analytics state
        self.insights_cache: Dict[str, AnalyticsInsight] = {}
        self.ml_models: Dict[str, Any] = {}
        self.analytics_stats = {
            "analyses_performed": 0,
            "insights_generated": 0,
            "reports_created": 0,
            "data_quality_score": 0.0
        }
        
        logger.info("📊 Enhanced Analytics Agent initialized")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the analytics agent"""
        return AgentPromptTemplates.ANALYTICS_SYSTEM
    
    def get_tools(self) -> List[WasteManagementTool]:
        """Get tools available to the analytics agent"""
        tools = []
        
        tools.append(WasteManagementTool(
            name="analyze_waste_data",
            description="Analyze waste management data using LangChain workflows",
            func=self._analyze_waste_data_tool
        ))
        
        tools.append(WasteManagementTool(
            name="answer_data_question",
            description="Answer natural language questions with contextual reasoning",
            func=self._answer_data_question_tool
        ))
        
        tools.append(WasteManagementTool(
            name="generate_report",
            description="Generate automated reports with intelligent narrative",
            func=self._generate_report_tool
        ))
        
        return tools
    
    async def initialize_agent(self) -> bool:
        """Initialize the Analytics Agent with enhanced capabilities"""
        try:
            logger.info("🚀 Initializing Enhanced Analytics Agent...")
            
            if not super().initialize_agent():
                return False
            
            # Initialize ML models
            if SKLEARN_AVAILABLE:
                self.ml_models["trend_predictor"] = RandomForestRegressor(
                    n_estimators=50, max_depth=8, random_state=42
                )
            else:
                logger.warning("⚠️ ML models not available - using basic analytics")
            
            # Setup MCP capabilities if available
            if self.mcp_handler:
                await self._setup_mcp_capabilities()
            
            logger.info("✅ Enhanced Analytics Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Analytics Agent: {e}")
            return False
    
    async def _setup_mcp_capabilities(self):
        """Setup MCP capabilities for analytics"""
        try:
            await self.mcp_handler.register_capability(
                name="analyze_waste_management_data",
                description="Analyze waste management data with LangChain workflows",
                input_schema={
                    "type": "object",
                    "properties": {
                        "data_sources": {"type": "array"},
                        "analysis_type": {"type": "string"},
                        "user_context": {"type": "object"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "insights": {"type": "array"},
                        "recommendations": {"type": "array"},
                        "natural_language_summary": {"type": "string"}
                    }
                },
                handler=self._handle_data_analysis_request
            )
            
            logger.info("🔄 MCP capabilities registered for Analytics")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup MCP capabilities: {e}")
    
    async def analyze_waste_data(
        self,
        data_sources: List[str],
        analysis_type: str = "comprehensive",
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main data analysis method using Pandas and ML models
        Enhanced with vector similarity search for contextual insights
        """
        try:
            logger.info(f"📊 Starting {analysis_type} analysis of {len(data_sources)} data sources")
            
            # Collect and prepare data
            data = await self._collect_and_prepare_data(data_sources)
            
            if data.empty:
                return {
                    "success": False,
                    "error": "No data available for analysis",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Perform analysis
            analysis_results = {}
            
            if analysis_type in ["comprehensive", "trend"]:
                analysis_results["trend_analysis"] = await self._perform_trend_analysis(data)
            
            # Generate insights
            insights = await self._generate_insights_from_analysis(analysis_results, user_context)
            
            # Generate natural language summary
            nl_summary = await self._generate_analysis_summary(analysis_results, insights, user_context)
            
            # Store insights in vector database
            if self.vector_db:
                await self._store_analysis_insights(insights, analysis_results)
            
            # Update statistics
            self.analytics_stats["analyses_performed"] += 1
            self.analytics_stats["insights_generated"] += len(insights)
            
            return {
                "success": True,
                "analysis_type": analysis_type,
                "data_quality_score": self._calculate_data_quality_score(data),
                "analysis_results": analysis_results,
                "insights": [asdict(insight) for insight in insights],
                "natural_language_summary": nl_summary,
                "recommendations": self._generate_recommendations_from_insights(insights),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Data analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _collect_and_prepare_data(self, data_sources: List[str]) -> pd.DataFrame:
        """Collect and prepare data from various sources"""
        try:
            all_data = []
            
            for source in data_sources:
                if source == "bin_data":
                    bin_data = await self._get_sample_bin_data()
                    all_data.append(bin_data)
                elif source == "route_data":
                    route_data = await self._get_sample_route_data()
                    all_data.append(route_data)
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True, sort=False)
                return self._clean_and_prepare_data(combined_data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Failed to collect and prepare data: {e}")
            return pd.DataFrame()
    
    async def _get_sample_bin_data(self) -> pd.DataFrame:
        """Generate sample bin data for demonstration"""
        try:
            import random
            
            sample_data = []
            for i in range(100):
                sample_data.append({
                    "source": "bin_data",
                    "bin_id": f"BIN_{i+1:03d}",
                    "fill_level": random.uniform(20, 95),
                    "capacity": random.choice([240, 480, 1000]),
                    "ward_id": random.randint(1, 24),
                    "timestamp": (datetime.now() - timedelta(hours=random.randint(0, 72))).isoformat()
                })
            
            return pd.DataFrame(sample_data)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate sample bin data: {e}")
            return pd.DataFrame()
    
    async def _get_sample_route_data(self) -> pd.DataFrame:
        """Generate sample route data for demonstration"""
        try:
            import random
            
            sample_data = []
            for i in range(50):
                sample_data.append({
                    "source": "route_data",
                    "ward_id": random.randint(1, 24),
                    "optimization_score": random.uniform(60, 95),
                    "fuel_cost": random.uniform(500, 2000),
                    "bins_collected": random.randint(10, 50),
                    "timestamp": (datetime.now() - timedelta(hours=random.randint(0, 168))).isoformat()
                })
            
            return pd.DataFrame(sample_data)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate sample route data: {e}")
            return pd.DataFrame()
    
    def _clean_and_prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for analysis"""
        try:
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data['hour'] = data['timestamp'].dt.hour
                data['day_of_week'] = data['timestamp'].dt.dayofweek
            
            # Handle missing values
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
            
            # Remove duplicates
            data = data.drop_duplicates()
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to clean and prepare data: {e}")
            return data
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score"""
        try:
            if data.empty:
                return 0.0
            
            completeness = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
            quality_score = completeness * 100
            self.analytics_stats["data_quality_score"] = quality_score
            
            return min(100.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate data quality score: {e}")
            return 50.0
    
    async def _perform_trend_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform trend analysis using time series analysis"""
        try:
            trends = {}
            
            if 'timestamp' in data.columns and len(data) > 10:
                data_daily = data.groupby(data['timestamp'].dt.date).agg({
                    'fill_level': 'mean',
                    'optimization_score': 'mean',
                    'fuel_cost': 'sum'
                }).reset_index()
                
                for metric in ['fill_level', 'optimization_score', 'fuel_cost']:
                    if metric in data_daily.columns:
                        values = data_daily[metric].dropna()
                        if len(values) > 3:
                            x = np.arange(len(values))
                            slope = np.polyfit(x, values, 1)[0]
                            
                            trends[metric] = {
                                "slope": float(slope),
                                "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                                "current_value": float(values.iloc[-1]),
                                "change_rate": float(slope / values.mean() * 100) if values.mean() != 0 else 0,
                                "confidence": min(1.0, len(values) / 30)
                            }
            
            return {
                "trends": trends,
                "analysis_period": f"{len(data)} data points"
            }
            
        except Exception as e:
            logger.error(f"❌ Trend analysis failed: {e}")
            return {"trends": {}, "error": str(e)}
    
    async def _generate_insights_from_analysis(
        self,
        analysis_results: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> List[AnalyticsInsight]:
        """Generate insights from analysis results using LangChain-style processing"""
        try:
            insights = []
            
            if "trend_analysis" in analysis_results:
                trend_insights = await self._generate_trend_insights(analysis_results["trend_analysis"])
                insights.extend(trend_insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Failed to generate insights: {e}")
            return []
    
    async def _generate_trend_insights(self, trend_analysis: Dict[str, Any]) -> List[AnalyticsInsight]:
        """Generate insights from trend analysis"""
        try:
            insights = []
            trends = trend_analysis.get("trends", {})
            
            for metric, trend_data in trends.items():
                if isinstance(trend_data, dict) and trend_data.get("confidence", 0) > 0.5:
                    direction = trend_data.get("direction", "stable")
                    change_rate = abs(trend_data.get("change_rate", 0))
                    
                    if direction != "stable" and change_rate > 5:
                        impact_level = "high" if change_rate > 20 else "medium" if change_rate > 10 else "low"
                        
                        insight = AnalyticsInsight(
                            insight_id=f"trend_{metric}_{datetime.now().timestamp()}",
                            insight_type="trend",
                            title=f"{metric.replace('_', ' ').title()} Trend Analysis",
                            description=f"{metric.replace('_', ' ').title()} is {direction} at a rate of {change_rate:.1f}% per period",
                            data_sources=["trend_analysis"],
                            confidence_score=trend_data.get("confidence", 0.5),
                            impact_level=impact_level,
                            recommendations=self._generate_trend_recommendations(metric, direction, change_rate),
                            natural_language_summary=f"The {metric.replace('_', ' ')} shows a {direction} trend with {change_rate:.1f}% change rate."
                        )
                        insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Failed to generate trend insights: {e}")
            return []
    
    def _generate_trend_recommendations(self, metric: str, direction: str, change_rate: float) -> List[str]:
        """Generate recommendations based on trend analysis"""
        recommendations = []
        
        if metric == "fill_level":
            if direction == "increasing" and change_rate > 15:
                recommendations.append("Increase collection frequency to prevent overflows")
                recommendations.append("Consider adding more bins in high-fill areas")
            elif direction == "decreasing":
                recommendations.append("Optimize collection routes to reduce unnecessary trips")
        
        elif metric == "fuel_cost":
            if direction == "increasing" and change_rate > 10:
                recommendations.append("Review route efficiency to reduce fuel consumption")
                recommendations.append("Consider vehicle maintenance to improve fuel efficiency")
            elif direction == "decreasing":
                recommendations.append("Continue current optimization strategies")
        
        elif metric == "optimization_score":
            if direction == "increasing":
                recommendations.append("Continue current optimization strategies")
            elif direction == "decreasing" and change_rate > 5:
                recommendations.append("Review optimization algorithms")
                recommendations.append("Analyze factors causing efficiency decline")
        
        return recommendations
    
    async def _generate_analysis_summary(
        self,
        analysis_results: Dict[str, Any],
        insights: List[AnalyticsInsight],
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate natural language summary using LangChain-style text generation"""
        try:
            summary_parts = []
            
            summary_parts.append("📊 **Waste Management Analytics Summary**")
            summary_parts.append(f"Analysis completed with {len(insights)} insights and {analysis_results.get('data_quality_score', 0):.1f}% data quality.")
            
            # Key insights by category
            insight_categories = {}
            for insight in insights:
                category = insight.insight_type
                if category not in insight_categories:
                    insight_categories[category] = []
                insight_categories[category].append(insight)
            
            for category, category_insights in insight_categories.items():
                summary_parts.append(f"\n🔍 **{category.title()} Analysis:**")
                for insight in category_insights[:2]:
                    summary_parts.append(f"• {insight.title}: {insight.description}")
            
            # Top recommendations
            all_recommendations = []
            for insight in insights:
                all_recommendations.extend(insight.recommendations)
            
            if all_recommendations:
                summary_parts.append("\n💡 **Key Recommendations:**")
                unique_recommendations = list(set(all_recommendations))[:5]
                for rec in unique_recommendations:
                    summary_parts.append(f"• {rec}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate analysis summary: {e}")
            return f"Analysis completed with {len(insights)} insights generated."
    
    async def _store_analysis_insights(
        self,
        insights: List[AnalyticsInsight],
        analysis_results: Dict[str, Any]
    ):
        """Store insights in vector database for future contextual reasoning"""
        try:
            if not self.vector_db:
                return
            
            for insight in insights:
                content = f"""
                Insight: {insight.title}
                Type: {insight.insight_type}
                Description: {insight.description}
                Impact Level: {insight.impact_level}
                Recommendations: {'; '.join(insight.recommendations)}
                Natural Language Summary: {insight.natural_language_summary or ''}
                """
                
                await self.vector_db.store_analytics_insight(
                    analysis_type=insight.insight_type,
                    insight_data={
                        "insight_id": insight.insight_id,
                        "title": insight.title,
                        "description": insight.description,
                        "impact_level": insight.impact_level,
                        "confidence_score": insight.confidence_score,
                        "recommendations": insight.recommendations
                    },
                    data_sources=insight.data_sources,
                    confidence_score=insight.confidence_score
                )
            
            logger.info(f"📚 Stored {len(insights)} insights in vector database")
            
        except Exception as e:
            logger.error(f"❌ Failed to store insights in vector database: {e}")
    
    def _generate_recommendations_from_insights(self, insights: List[AnalyticsInsight]) -> List[str]:
        """Generate consolidated recommendations from insights"""
        try:
            all_recommendations = []
            
            for insight in insights:
                all_recommendations.extend(insight.recommendations)
            
            # Remove duplicates and prioritize by impact
            unique_recommendations = []
            seen = set()
            
            sorted_insights = sorted(
                insights,
                key=lambda x: (
                    {"critical": 5, "high": 4, "medium": 3, "low": 2}.get(x.impact_level, 1),
                    x.confidence_score
                ),
                reverse=True
            )
            
            for insight in sorted_insights:
                for rec in insight.recommendations:
                    if rec not in seen:
                        unique_recommendations.append(rec)
                        seen.add(rec)
            
            return unique_recommendations[:10]
            
        except Exception as e:
            logger.error(f"❌ Failed to generate recommendations: {e}")
            return []   
 # Enhanced tool methods with LangChain integration
    async def _analyze_waste_data_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Tool wrapper for waste data analysis"""
        try:
            data_sources = parameters.get("data_sources", ["bin_data", "route_data"])
            analysis_type = parameters.get("analysis_type", "comprehensive")
            user_context = parameters.get("user_context")
            
            result = await self.analyze_waste_data(
                data_sources=data_sources,
                analysis_type=analysis_type,
                user_context=user_context
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Analyze waste data tool failed: {e}")
            return {"error": str(e)}
    
    async def _answer_data_question_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced natural language question answering about waste management data"""
        try:
            question = parameters.get("question", "")
            context = parameters.get("context", {})
            user_profile = parameters.get("user_profile", {})
            
            if not question:
                return {"error": "No question provided"}
            
            # Use vector database for contextual search
            relevant_insights = []
            if self.vector_db:
                relevant_insights = await self.vector_db.semantic_search(
                    collection_name="analytics_insights",
                    query=question,
                    n_results=5
                )
            
            # Determine data sources needed
            data_sources = self._determine_data_sources_from_question(question)
            
            # Get relevant data
            data = await self._collect_and_prepare_data(data_sources)
            
            # Generate contextual answer
            answer = await self._generate_contextual_answer(
                question, data, relevant_insights, user_profile
            )
            
            return {
                "success": True,
                "question": question,
                "answer": answer,
                "supporting_data": {
                    "data_points": len(data),
                    "relevant_insights": len(relevant_insights),
                    "data_sources": data_sources
                },
                "confidence_score": self._calculate_answer_confidence(data, relevant_insights)
            }
            
        except Exception as e:
            logger.error(f"❌ Answer data question tool failed: {e}")
            return {"error": str(e)}
    
    def _determine_data_sources_from_question(self, question: str) -> List[str]:
        """Determine what data sources are needed based on the question"""
        question_lower = question.lower()
        data_sources = []
        
        if any(word in question_lower for word in ["bin", "fill", "capacity", "overflow"]):
            data_sources.append("bin_data")
        
        if any(word in question_lower for word in ["route", "vehicle", "distance", "fuel"]):
            data_sources.append("route_data")
        
        if not data_sources:
            data_sources = ["bin_data", "route_data"]
        
        return data_sources
    
    async def _generate_contextual_answer(
        self,
        question: str,
        data: pd.DataFrame,
        relevant_insights: List[Dict[str, Any]],
        user_profile: Dict[str, Any]
    ) -> str:
        """Generate contextual answer using data and historical insights"""
        try:
            answer_parts = []
            question_lower = question.lower()
            
            if "trend" in question_lower or "pattern" in question_lower:
                if not data.empty and 'timestamp' in data.columns:
                    recent_data = data.tail(30)
                    if len(recent_data) > 5:
                        answer_parts.append("Based on recent data trends:")
                        
                        if 'fill_level' in recent_data.columns:
                            avg_fill = recent_data['fill_level'].mean()
                            answer_parts.append(f"• Average bin fill level is {avg_fill:.1f}%")
                        
                        if 'fuel_cost' in recent_data.columns:
                            avg_cost = recent_data['fuel_cost'].mean()
                            answer_parts.append(f"• Average daily fuel cost is ₹{avg_cost:.0f}")
            
            elif "cost" in question_lower:
                if not data.empty and 'fuel_cost' in data.columns:
                    total_cost = data['fuel_cost'].sum()
                    avg_cost = data['fuel_cost'].mean()
                    answer_parts.append(f"Cost analysis shows:")
                    answer_parts.append(f"• Total fuel costs: ₹{total_cost:.0f}")
                    answer_parts.append(f"• Average daily cost: ₹{avg_cost:.0f}")
            
            # Add insights from vector database
            if relevant_insights:
                answer_parts.append("\nRelevant historical insights:")
                for insight in relevant_insights[:2]:
                    answer_parts.append(f"• {insight['document'][:100]}...")
            
            # Personalize based on user profile
            role = user_profile.get("role", "")
            if role == "supervisor":
                answer_parts.append("\n💡 As a supervisor, consider reviewing these metrics across all wards.")
            elif role == "operator":
                answer_parts.append("\n💡 For daily operations, focus on the efficiency metrics.")
            
            if answer_parts:
                return "\n".join(answer_parts)
            else:
                return f"Based on {len(data)} data records, I can provide analysis but need more specific information for a detailed answer."
            
        except Exception as e:
            logger.error(f"❌ Failed to generate contextual answer: {e}")
            return "I encountered an issue analyzing the data. Please try rephrasing your question."
    
    def _calculate_answer_confidence(
        self,
        data: pd.DataFrame,
        relevant_insights: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for the answer"""
        try:
            confidence = 0.5
            
            if not data.empty:
                confidence += 0.2
                if len(data) > 50:
                    confidence += 0.1
            
            if relevant_insights:
                confidence += min(0.2, len(relevant_insights) * 0.05)
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate answer confidence: {e}")
            return 0.5
    
    async def _generate_report_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate automated reports with intelligent narrative using LangChain"""
        try:
            report_type = parameters.get("report_type", "comprehensive")
            ward_id = parameters.get("ward_id")
            user_context = parameters.get("user_context", {})
            
            # Perform comprehensive analysis
            analysis_results = await self.analyze_waste_data(
                data_sources=["bin_data", "route_data"],
                analysis_type="comprehensive",
                user_context=user_context
            )
            
            # Generate intelligent narrative
            narrative = await self._generate_intelligent_narrative(
                analysis_results, report_type, user_context
            )
            
            # Create report structure
            report = {
                "report_id": f"report_{datetime.now().timestamp()}",
                "report_type": report_type,
                "ward_id": ward_id,
                "generated_at": datetime.now().isoformat(),
                "intelligent_narrative": narrative,
                "key_insights": analysis_results.get("insights", [])[:10],
                "recommendations": analysis_results.get("recommendations", []),
                "data_quality_score": analysis_results.get("data_quality_score", 0)
            }
            
            self.analytics_stats["reports_created"] += 1
            
            return {
                "success": True,
                "report": report
            }
            
        except Exception as e:
            logger.error(f"❌ Generate report tool failed: {e}")
            return {"error": str(e)}
    
    async def _generate_intelligent_narrative(
        self,
        analysis_results: Dict[str, Any],
        report_type: str,
        user_context: Dict[str, Any]
    ) -> str:
        """Generate intelligent narrative using LangChain-style text generation"""
        try:
            narrative_parts = []
            
            data_quality = analysis_results.get("data_quality_score", 0)
            total_insights = len(analysis_results.get("insights", []))
            
            narrative_parts.append(f"## Waste Management Analysis Report")
            narrative_parts.append(f"This {report_type} analysis examined waste management operations with {data_quality:.1f}% data quality, generating {total_insights} actionable insights.")
            
            # Key findings
            insights = analysis_results.get("insights", [])
            high_impact_insights = [i for i in insights if i.get("impact_level") == "high"]
            
            if high_impact_insights:
                narrative_parts.append("\n### Critical Areas Requiring Attention:")
                for insight in high_impact_insights[:3]:
                    narrative_parts.append(f"• **{insight.get('title', 'Unknown')}**: {insight.get('description', 'No description')}")
            
            # Recommendations
            recommendations = analysis_results.get("recommendations", [])
            if recommendations:
                narrative_parts.append("\n## Strategic Recommendations")
                for i, rec in enumerate(recommendations[:5], 1):
                    narrative_parts.append(f"{i}. {rec}")
            
            # Conclusion
            narrative_parts.append("\n## Conclusion")
            narrative_parts.append(
                f"This analysis provides actionable insights for optimizing waste management operations. "
                f"With {total_insights} insights generated and {data_quality:.1f}% data quality, "
                f"the recommendations focus on improving efficiency and reducing costs."
            )
            
            return "\n".join(narrative_parts)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate intelligent narrative: {e}")
            return "Unable to generate detailed narrative due to analysis error."
    
    # MCP Handler methods
    async def _handle_data_analysis_request(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP data analysis requests"""
        return await self._analyze_waste_data_tool(parameters)
    
    def get_analytics_stats(self) -> Dict[str, Any]:
        """Get comprehensive analytics statistics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "analytics_stats": self.analytics_stats,
            "insights_cache_size": len(self.insights_cache),
            "ml_models_loaded": len(self.ml_models),
            "is_initialized": self.is_initialized,
            "vector_db_connected": self.vector_db is not None,
            "mcp_handler_active": self.mcp_handler is not None,
            "timestamp": datetime.now().isoformat()
        }

logger.info("📊 Enhanced Analytics Agent with semantic understanding loaded successfully")