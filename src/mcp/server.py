import os
from mcp.server.fastmcp import FastMCP
from ..harness.state import StateStore

mcp=FastMCP("trading-agent")
store=StateStore(os.getenv("TRADING_STATE_PATH","logs/state.json"))

@mcp.tool()
def get_status()->dict:
    """Read current trading harness state."""
    return store.load().__dict__

@mcp.tool()
def pause_trading()->dict:
    """Pause the harness. Does not place or cancel live orders."""
    s=store.load(); s.paused=True; store.save(s); return {"ok":True,"state":s.__dict__}

@mcp.tool()
def emergency_stop()->dict:
    """Set the persistent trading state to paused for emergency handling."""
    s=store.load(); s.paused=True; store.save(s); return {"ok":True,"emergency_stop":True,"state":s.__dict__}

if __name__=="__main__": mcp.run(transport="stdio")
