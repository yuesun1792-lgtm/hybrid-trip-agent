from backend.app.models import TripPlan


class ConversationMemory:
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.messages: list[dict[str, str]] = []
        self.extracted_fields: dict = {}
        self.last_plan: TripPlan | None = None
        self.rejected_destinations: list[str] = []
        self.current_stage: str = "new"

    def context(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "current_stage": self.current_stage,
            "last_plan_exists": self.last_plan is not None,
            "last_plan_mode": self.last_plan.planner_mode if self.last_plan else "",
            "last_plan_days": len(self.last_plan.itinerary) if self.last_plan else 0,
            "last_plan_destination": (
                self.last_plan.itinerary[0].destination
                if self.last_plan and self.last_plan.itinerary
                else ""
            ),
            "rejected_destinations": self.rejected_destinations,
            "recent_messages": self.messages[-6:],
        }


_CONVERSATIONS: dict[str, ConversationMemory] = {}


def get_conversation(conversation_id: str) -> ConversationMemory:
    if conversation_id not in _CONVERSATIONS:
        _CONVERSATIONS[conversation_id] = ConversationMemory(conversation_id)
    return _CONVERSATIONS[conversation_id]
