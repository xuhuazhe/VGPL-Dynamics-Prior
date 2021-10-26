# import torch.nn.functional as F
import torch
#

logits = torch.FloatTensor([[1,1,-1000]])
print(logits)
# # Sample soft categorical using reparametrization trick:
print(torch.nn.functional.gumbel_softmax(logits, tau=1, hard=True))
# Sample hard categorical usi
# ng "Straight-through" trick:
# F.gumbel_softmax(logits, tau=1, hard=True)