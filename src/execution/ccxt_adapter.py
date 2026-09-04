import ccxt
from .exchange import ExchangeAdapter
class CCXTSpotAdapter(ExchangeAdapter):
    def __init__(self,exchange_id="binance",api_key=None,secret=None,enable_live=False):
        if not enable_live: raise RuntimeError("Live execution is disabled. Explicitly enable it in production.")
        cls=getattr(ccxt,exchange_id); self.ex=cls({"apiKey":api_key,"secret":secret,"enableRateLimit":True,"options":{"defaultType":"spot"}}); self.ex.load_markets()
    def balance(self): return self.ex.fetch_balance()
    def ticker(self,symbol): return self.ex.fetch_ticker(symbol)
    def target_position(self,symbol,target): raise NotImplementedError("Use OrderManager after adding exchange-specific precision/order verification.")
    def cancel_all(self,symbol=None):
        for o in self.ex.fetch_open_orders(symbol): self.ex.cancel_order(o["id"],symbol)
