class OrderManager:
    """Execution planner. Live order placement must remain behind SafetyLayer."""
    def __init__(self,adapter,safety): self.adapter=adapter; self.safety=safety
    def plan(self, current_position, target_position, portfolio_value):
        delta=float(target_position)-float(current_position)
        return {"delta_position":delta,"notional":abs(delta)*portfolio_value}
