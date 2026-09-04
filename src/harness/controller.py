class TradingHarness:
    def __init__(self,model,feature_engine,safety,state_store,execution=None): self.model=model; self.features=feature_engine; self.safety=safety; self.state_store=state_store; self.execution=execution
    def status(self): return self.state_store.load()
    def pause(self): s=self.state_store.load(); s.paused=True; self.state_store.save(s)
    def emergency_stop(self): self.safety.emergency_stop(); self.pause();
