class SafetyLayer:
    def __init__(self,max_order_fraction=.25,max_daily_loss=.05): self.max_order_fraction=max_order_fraction; self.max_daily_loss=max_daily_loss; self.enabled=True
    def validate(self,portfolio_value,order_value,daily_return,price_age_seconds,max_price_age=120):
        if not self.enabled: return False,"safety disabled"
        if price_age_seconds>max_price_age: return False,"stale market data"
        if order_value>portfolio_value*self.max_order_fraction: return False,"order too large"
        if daily_return<=-self.max_daily_loss: return False,"daily loss limit"
        return True,"ok"
    def emergency_stop(self): self.enabled=False
