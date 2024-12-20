import torch
import torch.nn as nn

class SequentialDropout(nn.Module):
    def __init__(self, min_p: float = 0.05,scale_output: bool= True):
        super(SequentialDropout, self).__init__()
        if min_p < 0 or min_p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.min_p = min_p
        self.scale_output = scale_output
        #device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    # only set up to handle 1D tensors currently    
    def forward(self, X):
        sh = X.shape
  
        if len(sh) != 2:# 
            raise ValueError("shape expected to be (_,N) " "but got {}".format(sh))
            
        if self.training:
            # mask generation
            sh = sh[1]
            max_split = int(sh * self.min_p)
            split = torch.randint(0, sh - max_split, (1,), device=self.device) 
            ones = torch.ones(sh - split, device=self.device)
            zeros = torch.zeros(split, device=self.device)
            
            mask = torch.cat((ones,zeros), 0)
            mask = mask.repeat(X.shape[0],1)
            # scale by dsitribution
            scale = 1.
            if self.scale_output:
                #scale =  float(sh) / (sh-split)
                
                #calculate keep prob
                #scale = (sh - split) / float(sh)

                orig_sum = torch.sum(X)
                new_sum = torch.sum(X * mask)
                scale = orig_sum / new_sum
                #print(orig_sum, torch.sum(X * mask * scale))

             
            
            #print("break")
            #print(scale)
            #print(X * mask )
            #print(X * mask * scale)
            
            return (X * mask) * scale
        
            
        return X