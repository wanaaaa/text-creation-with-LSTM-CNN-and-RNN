from funClass import *
import torch.nn as nn


N_INPUT = 5
N_NEURONS = 3

X0_batch = torch.tensor([[0,1,2,0, 1], [3,4,5,0, 1],
                         [6,7,8,0, 1], [9,0,1,0, 1]],
                        dtype = torch.float) #t=0 => 4 X 4

X1_batch = torch.tensor([[9,8,7,0, 1], [0,0,0,0, 1],
                         [6,5,4,0, 1], [3,2,1,0, 1]],
                        dtype = torch.float) #t=1 => 4 X 4

model = SimpleRNN(N_INPUT, N_NEURONS)

Y0_val, Y1_val = model(X0_batch, X1_batch)

print("-->", Y0_val)
print("-->", Y1_val)
# ===================================================================
# ===================================================================

FIXED_BATCH_SIZE = 4 # our batch size is fixed for now
N_INPUT = 3
N_NEURONS = 5

X_batch = torch.tensor([[[0,1,2], [3,4,5],
                         [6,7,8], [9,0,1]],
                        [[9,8,7], [0,0,0],
                         [6,5,4], [3,2,1]]
                       ], dtype = torch.float) # X0 and X1


model = CleanBasicRNN(FIXED_BATCH_SIZE, N_INPUT, N_NEURONS)
output_val, states_val = model(X_batch)
print(output_val) # contains all output for all timesteps
print(states_val) # contains values for final state or final timestep, i.e., t=1





















