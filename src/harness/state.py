from dataclasses import dataclass,asdict
import json,os
@dataclass
class TradingState:
    symbol:str=""; position:float=0.; portfolio_value:float=0.; last_action:float=0.; last_timestamp:str=""; paused:bool=True
class StateStore:
    def __init__(self,path="logs/state.json"): self.path=path
    def load(self):
        if not os.path.exists(self.path): return TradingState()
        return TradingState(**json.load(open(self.path)))
    def save(self,state): os.makedirs(os.path.dirname(self.path),exist_ok=True); json.dump(asdict(state),open(self.path,"w"),indent=2)
