import numpy as np
import torch



class TCNBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, dilation, activation=True):
    super().__init__()
    self.conv = torch.nn.Conv1d(
        in_channels, 
        out_channels, 
        kernel_size, 
        dilation=dilation, 
        padding=0, #((kernel_size-1)//2)*dilation,
        bias=True)
    torch.nn.init.xavier_uniform_(self.conv.weight) # default is kaiming_uniform_(self.res.weight a=math.sqrt(5)) this equals U(−k,k), where k=groups/(Cin∗kernel_size) (cf. source code), but keras uses GlorotUniform which is xavier_uniform_; for timbral fx kaiming_uniform_ with! a=math.sqrt(5) seems to work better
    torch.nn.init.zeros_(self.conv.bias) # also default is kaiming_uniform_(self.res.bias, a=math.sqrt(5)), see source code for implementation with small inchannels, but keras uses Zeros for bias
    if activation:
      # self.act = torch.nn.Tanh()
      self.act = torch.nn.PReLU()
    # this is the residual connection with 1x1 conv to match the channels and mix the desired amount of residual in
    self.res = torch.nn.Conv1d(in_channels, out_channels, 1, bias=False)
    torch.nn.init.xavier_uniform_(self.res.weight)
    self.kernel_size = kernel_size
    self.dilation = dilation

  def forward(self, x):
    x_in = x
    x = self.conv(x)
    if hasattr(self, "act"):
      x = self.act(x)
    x_res = self.res(x_in)
    x_res = x_res[..., (self.kernel_size-1)*self.dilation:]
    x = x + x_res
    return x

class TCN(torch.nn.Module):
  def __init__(self, n_inputs=1, n_outputs=1, n_blocks=10, kernel_size=13, n_channels=64, dilation_growth=4):
    super().__init__()
    self.kernel_size = kernel_size
    self.n_channels = n_channels
    self.dilation_growth = dilation_growth
    self.n_blocks = n_blocks
    self.stack_size = n_blocks

    self.blocks = torch.nn.ModuleList()
    for n in range(n_blocks):
      if n == 0:
        in_ch = n_inputs
        out_ch = n_channels
        act = True
      elif (n+1) == n_blocks:
        in_ch = n_channels
        out_ch = n_outputs
        act = True
      else:
        in_ch = n_channels
        out_ch = n_channels
        act = True
      
      dilation = dilation_growth ** n
      self.blocks.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, activation=act))

  def forward(self, x):
    for block in self.blocks:
      x = block(x)
    return x
  
  def compute_receptive_field(self):
    """Compute the receptive field in samples."""
    rf = self.kernel_size
    for n in range(1, self.n_blocks):
        dilation = self.dilation_growth ** (n % self.stack_size)
        rf = rf + ((self.kernel_size - 1) * dilation)
    return rf

class TCNDiscriminator(torch.nn.Module):
    def __init__(self, n_inputs=1, n_outputs=1, n_blocks=10, kernel_size=13, n_channels=64, dilation_growth=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.dilation_growth = dilation_growth
        self.n_blocks = n_blocks

        self.blocks = torch.nn.ModuleList()
        for n in range(n_blocks):
            if n == 0:
                in_ch = n_inputs
                out_ch = n_channels
                act = True
            else:
                in_ch = n_channels
                out_ch = n_channels
                act = True
            
            dilation = dilation_growth ** n
            self.blocks.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, activation=act))

        self.global_avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(n_channels, n_outputs)
        # self.sigmoid = torch.nn.Sigmoid()
        # self.re= torch.nn.ReLU()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)  # remove the time dimension
        x = self.fc(x)
        # x = self.sigmoid(x)
        return x
