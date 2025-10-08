# ai_agents/ward_agent.py
class WardAgent:
    """Simple AI agent for ward insights."""

    def __init__(self, ward_id: int):
        self.ward_id = ward_id

    def predict_issue_trend(self):
        """Dummy trend prediction."""
        # Replace with real ML/AI logic later
        trend = "increasing" if self.ward_id % 2 == 0 else "stable"
        return {"ward_id": self.ward_id, "trend": trend}

    def recommend_actions(self):
        """Dummy recommendations."""
        actions = ["Inspect drains", "Check street lights", "Community awareness"]
        return {"ward_id": self.ward_id, "recommendations": actions}
