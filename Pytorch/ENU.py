from torch.nn import Linear, Parameter, Module
from torch import sigmoid, matmul, tanh, cat, clamp, stack, Tensor, randn, ones, zeros

class ENU(nn.Module):
    def __init__(self, input_channels_size, output_channels_size, batch_size=1, memory_state_size=4, 
                 inner_gate_hidden_units=7, output_gate_hidden_units=4):
        super(ENU, self).__init__()
        
        self.input_size = input_channels_size
        self.h_size = memory_state_size
        self.output_size = output_channels_size
        self.inner_gate_hidden_units = inner_gate_hidden_units
        self.output_gate_hidden_units = output_gate_hidden_units
        self.batch_size = batch_size
        
        # reset gate
        self.reset_gate = Linear(self.input_size + self.h_size + self.output_size, self.h_size)
        self.reset_gate.weights = randn(self.input_size + self.h_size + self.output_size, self.h_size, requires_grad=True)
        self.reset_gate.biass = randn(self.h_size, requires_grad=True)
        
        # update gate
        self.update_gate = Linear(self.input_size + self.h_size + self.output_size, self.h_size)
        self.reset_gate.weights = randn(self.input_size + self.h_size + self.output_size, self.h_size, requires_grad=True)
        self.reset_gate.biass = randn(self.h_size, requires_grad=True)        
        
        # cell gate
        self.cell_gate = Linear(self.input_size + self.h_size + self.output_size, self.h_size)
        self.reset_gate.weights = randn(self.input_size + self.h_size + self.output_size, self.h_size, requires_grad=True)
        self.reset_gate.biass = randn(self.h_size, requires_grad=True)
        
        # output gate
        self.output_gate = Linear(self.h_size, self.output_size)
        self.reset_gate.weights = randn(self.h_size, self.output_size, requires_grad=True)
        self.reset_gate.biass = randn(self.output_size, requires_grad=True)
        
        self.predict_gate = Linear(self.output_size, 1)
        self.predict_gate.weights = randn(self.output_size, 1, requires_grad=True)
        self.predict_gate.biass = randn(1, requires_grad=True)
        
        # pre memory sate and output
        self.h = randn(self.batch_size, 1, self.h_size)
        self.o = randn(self.batch_size, 1, self.output_size)
    
    def Reset_Gate(self, data):
        return self.reset_gate(data)

    def Update_Gate(self, data):
        return self.update_gate(data)

    def Cell_Gate(self, data):
        return self.cell_gate(data)

    def Output_Gate(self, data):
        return self.output_gate(data)
    
    def step(self, x): # input each SIze(1, 3)
        self.input = cat((self.h, self.o, x), -1) # Size (1, 10)

        # Reset Gate
        self.r = sigmoid(self.Reset_Gate(self.input)) # Size (1, 4)
        # Update Gate
        self.z = sigmoid(self.Update_Gate(self.input)) # Size (1, 4)
        
        # Cell Gate
        self.cell_gate_input = cat((self.r * self.h, self.o, x), -1) # Size(1, 10)
        self.h_bar = tanh(self.Cell_Gate(self.cell_gate_input)) # Size (1, 4)
        
        # Memory State
        self.h = (1 - self.z) * self.h + self.z * self.h_bar # new memory state 
        
        # Output Gate
        self.o = clamp(self.Output_Gate(self.h), 0, 1) # Size (1, 3)

        return self.o
    
    def forward(self, x):
        # x : (batch_size, sequence, input_channels)
        self.h = zeros(x.size()[0], 1, self.h_size)
        self.o = zeros(x.size()[0], 1, self.output_size)
        
        for i in range(x.size()[1]):
            output = self.step(x[:, i, :].view(self.batch_size, 1, self.input_size))
        
        return output
    
    def predict(self, output=None):
        if output is None:
            return self.predict_gate(self.output)
        else:
            return self.predict_gate(output)