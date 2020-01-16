from torch.nn import Linear, Parameter, Module
from torch import sigmoid, matmul, tanh, cat, clamp, stack, Tensor, randn, ones

class ENU(Module):
    def __init__(self, input_channels_size, output_channels_size, memory_state_size=4, 
                 inner_gate_hidden_units=7, output_gate_hidden_units=4):
        super(ENU, self).__init__()
        
        self.input_size = input_channels_size
        self.h_size = memory_state_size
        self.output_size = output_channels_size
        self.inner_gate_hidden_units = inner_gate_hidden_units
        self.output_gate_hidden_units = output_gate_hidden_units
        
        # reset gate
        self.reset_gate1 = Linear(self.input_size + self.h_size + self.output_size, self.inner_gate_hidden_units)
        self.reset_gate2 = Linear(self.inner_gate_hidden_units, self.h_size)
        
        # update gate
        self.update_gate1 = Linear(self.input_size + self.h_size + self.output_size, self.inner_gate_hidden_units)
        self.update_gate2 = Linear(self.inner_gate_hidden_units, self.h_size)
        
        # cell gate
        self.cell_gate1 = Linear(self.input_size + self.h_size + self.output_size, self.inner_gate_hidden_units)
        self.cell_gate2 = Linear(self.inner_gate_hidden_units, self.h_size)

        # output gate
        self.output_gate1 = Linear(self.h_size, self.output_gate_hidden_units)
        self.output_gate2 = Linear(self.output_gate_hidden_units, self.output_size)

        # pre memory sate and output
        self.h = randn(1, self.h_size)
        self.o = randn(1, self.output_size)
    
    def Reset_Gate(self, data):
        return self.reset_gate2(self.reset_gate1(data))

    def Update_Gate(self, data):
        return self.update_gate2(self.update_gate1(data))

    def Cell_Gate(self, data):
        return self.cell_gate2(self.cell_gate1(data))

    def Output_Gate(self, data):
        return self.output_gate2(self.output_gate1(data))
    
    def forward(self, x): # input each SIze(1, 3)
        self.input = cat((self.h, self.o, x), 1) # Size (1, 10)

        # Reset Gate
        self.r = sigmoid(self.Reset_Gate(self.input)) # Size (1, 4)
        # Update Gate
        self.z = sigmoid(self.Update_Gate(self.input)) # Size (1, 4)
        
        # Cell Gate
        self.cell_gate_input = cat((self.r * self.h, self.o, x), 1) # Size(1, 10)
        self.h_bar = tanh(self.Cell_Gate(self.cell_gate_input)) # Size (1, 4)
        
        # Memory State
        self.h = (1 - self.z) * self.h + self.z * self.h_bar # new memory state 
        
        # Output Gate
        self.o = clamp(self.Output_Gate(self.h), 0, 1) # Size (1, 3)

        return self.o