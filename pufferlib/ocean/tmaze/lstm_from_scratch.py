import torch




def sigmoid(x: torch.Tensor):
    return 1 / (1 + torch.exp(-x))

def tanh(x: torch.Tensor):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


class TMazePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 32
        self.is_continuous = False 
        self.lstm = LSTMCell(3,self.hidden_size)
        self.W_value = torch.nn.Parameter(torch.randn(self.hidden_size, 1))
        self.b_value = torch.nn.Parameter(torch.zeros(1))
        self.W_policy = torch.nn.Parameter(torch.randn(self.hidden_size, 3))
        self.b_policy = torch.nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor, state: dict[str, torch.Tensor]):
        h = state['lstm_h']
        c = state['lstm_c']
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
            c = torch.zeros(x.shape[0], self.hidden_size, device=x.device)

        if x.dim() == 3:
            # Training: x is (batch, time, obs_size)
            B, T, _ = x.shape
            all_h = []
            for t in range(T):
                h, c = self.lstm(x[:, t, :], h, c)
                all_h.append(h)
            h_seq = torch.stack(all_h, dim=1)  # (batch, time, hidden)
            h_flat = h_seq.reshape(B * T, self.hidden_size)
            logits = h_flat @ self.W_policy + self.b_policy
            value = (h_flat @ self.W_value + self.b_value).reshape(B, T)
        else:
            # Eval: x is (batch, obs_size)
            h, c = self.lstm(x, h, c)
            logits = h @ self.W_policy + self.b_policy
            value = h @ self.W_value + self.b_value

        state['lstm_h'] = h.detach()
        state['lstm_c'] = c.detach()
        return logits, value

    def forward_eval(self, x, state):
        return self.forward(x, state)

        


class LSTMCell(torch.nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        super().__init__()
        self.W_f = torch.nn.Parameter(torch.randn(hidden_size + input_size, hidden_size))
        self.b_f = torch.nn.Parameter(torch.zeros(hidden_size))
        self.W_i = torch.nn.Parameter(torch.randn(hidden_size + input_size, hidden_size))
        self.b_i = torch.nn.Parameter(torch.zeros(hidden_size))
        self.W_c = torch.nn.Parameter(torch.randn(hidden_size + input_size, hidden_size))
        self.b_c = torch.nn.Parameter(torch.zeros(hidden_size))
        self.W_o = torch.nn.Parameter(torch.randn(hidden_size + input_size, hidden_size))
        self.b_o = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, h_old: torch.Tensor, c_old: torch.Tensor):
        concattenated_input = torch.cat([x, h_old], dim=1)
        forget_out = sigmoid(concattenated_input @ self.W_f + self.b_f)
        input_out = sigmoid(concattenated_input @ self.W_i + self.b_i)
        candidate_out = tanh(concattenated_input @ self.W_c + self.b_c)
        output_out = sigmoid(concattenated_input @ self.W_o + self.b_o)
        c_new = (c_old * forget_out) + (input_out * candidate_out)
        h_new = tanh(c_new) * output_out
        return h_new, c_new
       

if __name__ == '__main__':                                                                                  
    cell = TMazePolicy()
                                                                                                            
    x = torch.tensor([1.0, 0.0, 0.0])  # signal=1, pos=0, not at junction                                   
    h = torch.zeros(32)                                                                                     
    c = torch.zeros(32)
    state = {
        'lstm_h': h,
        'lstm_c': c
    }                                                                               
                
    logits, value = cell.forward(x, state)                                                                    
    print(f"logits shape: {logits.shape}")   # should be (3,)                                                   
    print(f"value shape: {value.shape}")     # should be (1,)
    print(f"h updated: {state['lstm_h'].shape}")  # should be (32,)                                          
                                                                                                            
    # Test that gradients flow                                                                              
    loss = logits.sum() + value.sum()
    loss.backward()                                                                                         
    print(f"W_policy gradient exists: {cell.W_policy.grad is not None}")                                        
    print(f"LSTM W_f gradient exists: {cell.lstm.W_f.grad is not None}")     