import numpy as np
import scipy.signal
import math

import torch
import torchaudio
import torchinfo
import onnx

import os
import IPython
from tqdm import tqdm

import matplotlib.pyplot as plt
import librosa.display

if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"

name = 'model_0'


if not os.path.exists('models/'+name):
    os.makedirs('models/'+name)
else:
    print("A model with the same name already exists. Please choose a new name.")
    exit

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
  

#@title Use pre-loaded audio examples for steering
effect_type = "Reverb" #@param ["Compressor", "Reverb", "UltraTab", "Amp"]

if effect_type == "Compressor":
  input_file = "audio/drum_kit_clean.wav"
  output_file = "audio/drum_kit_comp_agg.wav"
elif effect_type == "Reverb":
  input_file = "audio/acgtr_clean.wav"
  output_file = "audio/acgtr_reverb.wav"
elif effect_type == "UltraTab":
  input_file = "audio/acgtr_clean.wav"
  output_file = "audio/acgtr_ultratab.wav"
elif effect_type == "Amp":
  input_file = "audio/ts9_test1_in_FP32.wav"
  output_file = "audio/ts9_test1_out_FP32.wav"

x, sample_rate = torchaudio.load(input_file)
y, sample_rate = torchaudio.load(output_file)

x = x[0:1,:]
y = y[0:1,:]

# print("x shape", x.shape)
# print(f"x = {x}")
# print("y shape", y.shape)
# print(f"y = {y}")

# print("input file", x.shape)
# IPython.display.display(IPython.display.Audio(data=x, rate=sample_rate))
# print("output file", y.shape)
# IPython.display.display(IPython.display.Audio(data=y, rate=sample_rate))


#@title TCN model training parameters
kernel_size = 13 #@param {type:"slider", min:3, max:32, step:1}
n_blocks = 4 #@param {type:"slider", min:2, max:30, step:1}
dilation_growth = 10 #@param {type:"slider", min:1, max:10, step:1}
n_channels = 32 #@param {type:"slider", min:1, max:128, step:1}
n_iters = 300 #@param {type:"slider", min:0, max:10000, step:1}
length = 508032 #@param {type:"slider", min:0, max:524288, step:1}
lr = 0.001 #@param {type:"number"}

# reshape the audio
x_batch = x.view(1,1,-1)
y_batch = y.view(1,1,-1)

# print(f"x_batch shape: {x_batch.shape}")
# print(f"y_batch shape: {y_batch.shape}")

# build the model
model = TCN(
    n_inputs=1,
    n_outputs=1,
    kernel_size=kernel_size, 
    n_blocks=n_blocks, 
    dilation_growth=dilation_growth, 
    n_channels=n_channels)
rf = model.compute_receptive_field()
params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Parameters: {params*1e-3:0.3f} k")
print(f"Receptive field: {rf} samples or {(rf/sample_rate)*1e3:0.1f} ms")

# setup loss function, optimizer, and scheduler
loss_fn_mse = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr)
ms1 = int(n_iters * 0.8)
ms2 = int(n_iters * 0.95)
milestones = [ms1, ms2]
print(
    "Learning rate schedule:",
    f"1:{lr:0.2e} ->",
    f"{ms1}:{lr*0.1:0.2e} ->",
    f"{ms2}:{lr*0.01:0.2e}",
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones,
    gamma=0.1,
    verbose=False,
)

summary = torchinfo.summary(model, (1, 1, 228308), device=device)
print(summary)

print("The reduction in the first layer is from the kernel size. The reduction in the subsequent layers is from the dilation factor and kernel size.")

# move tensors to GPU
if torch.cuda.is_available():
  model.to(device)
  x_batch = x_batch.to(device)
  y_batch = y_batch.to(device)

start_idx = rf 
stop_idx = start_idx + length

# the data is the same with every iteration
x_crop = x_batch[...,start_idx-rf+1:stop_idx]
y_crop = y_batch[...,start_idx:stop_idx]

print(f"x_crop = {x_crop.shape}")
print(f"y_crop = {y_crop.shape}")

########## iteratively update the weights

# this is only for the progress 
pbar = tqdm(range(n_iters))

for n in pbar:
  optimizer.zero_grad()

  y_hat = model(x_crop)
  loss = loss_fn_mse(y_hat, y_crop)

  loss.backward()
  optimizer.step()
  
  scheduler.step()
  if (n+1) % 1 == 0:
    pbar.set_description(f" Loss: {loss.item()} | ")

torch.save(model.state_dict(), "models/"+name+"/"+name+".pth")    

model.load_state_dict(torch.load("models/"+name+"/"+name+".pth", map_location=torch.device('cpu')))

# Run Prediction #################################################
# Test the model on the testing data #############################

# needed because in the train we crop the target
x_pad = torch.nn.functional.pad(x_batch, (rf-1, 0))

model.eval()
with torch.no_grad():
  y_hat = model(x_pad)

input = x_batch.view(-1).detach().cpu().numpy()[-y_hat.shape[-1]:]
output = y_hat.view(-1).detach().cpu().numpy()
target = y_batch.view(-1).detach().cpu().numpy()

# print(f"Input shape: {input.shape}")
# print(f"Output shape: {output.shape}")
# print(f"Target shape: {target.shape}")

# apply highpass to outpu to remove DC
sos = scipy.signal.butter(8, 20.0, fs=sample_rate, output="sos", btype="highpass")
output = scipy.signal.sosfilt(sos, output)

input /= np.max(np.abs(input))
output /= np.max(np.abs(output))
target /= np.max(np.abs(target))

# fig, ax = plt.subplots(nrows=1, sharex=True)
# librosa.display.waveshow(target, sr=sample_rate, color='b', alpha=0.5, ax=ax, label='Target')
# librosa.display.waveshow(output, sr=sample_rate, color='r', alpha=0.5, ax=ax, label='Output')

# print("Input (clean)")
# IPython.display.display(IPython.display.Audio(data=input, rate=sample_rate))
# print("Target")
# IPython.display.display(IPython.display.Audio(data=target, rate=sample_rate))
# print("Output")
# IPython.display.display(IPython.display.Audio(data=output, rate=sample_rate))
# plt.legend()
# plt.show(fig)

# Load and Preprocess Data ###########################################
x_whole, sample_rate = torchaudio.load("audio/piano_clean.wav")
x_whole = x_whole[0,:]
x_whole = x_whole.view(1,1,-1).to(device)

# Padding on both sides of the receptive field
x_whole = torch.nn.functional.pad(x_whole, (rf-1, rf-1))

with torch.no_grad():
  y_whole = model(x_whole)

x_whole = x_whole[..., -y_whole.shape[-1]:]

y_whole /= y_whole.abs().max()

# apply high pass filter to remove DC
sos = scipy.signal.butter(8, 20.0, fs=sample_rate, output="sos", btype="highpass")
y_whole = scipy.signal.sosfilt(sos, y_whole.cpu().view(-1).numpy())

x_whole = x_whole.view(-1).cpu().numpy()

y_whole = (y_whole * 0.8)
# IPython.display.display(IPython.display.Audio(data=x_whole, rate=sample_rate))
# IPython.display.display(IPython.display.Audio(data=y_whole, rate=sample_rate))

x_whole /= np.max(np.abs(x_whole))
y_whole /= np.max(np.abs(y_whole))

# fig, ax = plt.subplots(nrows=1, sharex=True)
# librosa.display.waveshow(y_whole, sr=sample_rate, color='r', alpha=0.5, ax=ax, label='Output')
# librosa.display.waveshow(x_whole, sr=sample_rate, alpha=0.5, ax=ax, label='Input', color="blue")
# plt.legend()
# plt.show(fig)

print("done")

import torch.optim as optim
from copy import deepcopy
# Define the Generator (TCN)
generator = (deepcopy(model)).to(device)
# Define the Discriminator
discriminator = TCNDiscriminator(
    n_inputs=1,
    n_outputs=1,
    kernel_size=13, 
    n_blocks=4, 
    dilation_growth=10, 
    n_channels=32)


summary = torchinfo.summary(discriminator, (1, 1, 226976), device=device)
print(summary)

# Define the Loss Functions
# adversarial_loss_fn = torch.nn.BCEWithLogitsLoss()
adversarial_loss_fn = torch.nn.MSELoss()

# Training Loop
num_epochs = 1000

ms1 = int(num_epochs * 0.8)  
ms2 = int(num_epochs * 0.95)  
milestones = [ms1, ms2]

# this is only for the progress 
pbar = tqdm(range(num_epochs))

g_learning_rate=0.0001
d_learning_rate=0.0001

generator_optimizer = optim.Adam(generator.parameters(), lr=g_learning_rate)
generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    generator_optimizer,
    milestones=milestones,
    gamma=0.1,
    verbose=False
)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=d_learning_rate)
discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    discriminator_optimizer,
    milestones=milestones,
    gamma=0.1,
    verbose=False
)

d_losses=[]
g_losses=[]
x_crop.to(device)
y_crop.to(device)


for epoch in pbar:
    # Generate fake data using the TCN (generator)
    discriminator.train()
    generator.train()
    # real_labels = torch.ones(1, 1).to(device)
    # fake_labels = torch.zeros(1, 1).to(device)
    
    discriminator_optimizer.zero_grad()
    with torch.no_grad():
        fake_data = generator(x_crop)
        fake_data.to(device)
    real_predictions = discriminator(y_crop)
    real_predictions.to(device)
    fake_predictions = discriminator(fake_data)
    fake_predictions.to(device)

    #distance
    distance = loss_fn_mse(fake_data, y_crop)

    # discriminator_loss = adversarial_loss_fn(real_predictions, real_labels) + adversarial_loss_fn(fake_predictions, fake_labels)
    discriminator_loss = torch.relu(1 - real_predictions) + torch.relu(fake_predictions)
    discriminator_loss.backward()
    d_losses.append(discriminator_loss.item())
    discriminator_optimizer.step()
    discriminator_scheduler.step()

    # Train the generator
    generator_optimizer.zero_grad()
    fake_data = generator(x_crop)
    fake_predictions = discriminator(fake_data)
    fake_predictions.to(device)

    # generator_loss = adversarial_loss_fn(fake_predictions, real_labels)
    generator_loss = -fake_predictions + distance
    generator_loss.backward()
    g_losses.append(generator_loss.item()) 
    generator_optimizer.step()
    generator_scheduler.step()
    if (n+1) % 1 == 0:
      pbar.set_description(f" GLoss: {generator_loss.item()} | DLoss: {discriminator_loss.item()} | loss: {loss_fn_mse(fake_data, y_crop)}")
    # print(f"Epoch {epoch} | Discriminator Loss: {discriminator_loss.item()} | Generator Loss: {generator_loss.item()}")
