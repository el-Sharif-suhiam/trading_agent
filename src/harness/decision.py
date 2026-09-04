import torch
class DecisionEngine:
    def __init__(self,model,device): self.model=model.eval(); self.device=device
    @torch.no_grad()
    def predict(self,window,hidden=None):
        x=torch.as_tensor(window,dtype=torch.float32,device=self.device).unsqueeze(0).unsqueeze(0)
        dist,value,hidden=self.model(x,hidden); mean=(dist.concentration1/(dist.concentration1+dist.concentration0)).item(); return {"target_position":mean,"value":value.item()},hidden
