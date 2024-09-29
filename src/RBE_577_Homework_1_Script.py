"""
RBE 577: Homework 1
Azzam Shaikh
Sept. 29, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

device =  "cpu" # GPU is available but ran into several issues when using GPU
print(device)

# Define random seeds to control randomness per run
torch.manual_seed(42)
np.random.seed(42)

# Define thruster command ranges
F1 = np.arange(-10000, 10000, 1)
F2 = np.arange(-5000, 5000, 1)
alpha2 = np.arange(-3.14, 3.14, 1)
F3 = np.arange(-5000, 5000, 1)
alpha3 = np.arange(-3.14, 3.14, 1)

# Function to simulate a random walk with clamping
def random_walk_process(initial_value, steps, step_size, min_val, max_val):
    """ 
    Generate a random walk. Clip values to ensure they dont exceed min and max range values 
    """
    # Initialize a list with the initial value
    random_walk = [initial_value]

    # Run the loop for 999424 steps
    for _ in range(steps-1):
        
        # Select either a positive or negative step
        step = np.random.uniform(-step_size, step_size)

        # Add the step to the last value
        new_value = random_walk[-1] + step
        
        # Ensure the value does not exceed the defined limits
        new_value = np.clip(new_value, min_val, max_val)

        # Add the new value to the list
        random_walk.append(new_value)

    # Return the list
    return random_walk


def apply_random_walk(F1_range, F2_range, F3_range, alpha2_range, alpha3_range, steps):
    """ 
    Create the u vector via a random walk
    """
    # Initialize a starting points for the random walk from the constraint ranges
    F1_init = np.random.choice(F1_range)
    F2_init = np.random.choice(F2_range)
    F3_init = np.random.choice(F3_range)
    alpha2_init = np.random.choice(alpha2_range)
    alpha3_init = np.random.choice(alpha3_range)

    # Perform a random walk for the 5 inputs
    F1_walk = random_walk_process(F1_init, steps, 100, F1_range[0], F1_range[-1])
    F2_walk = random_walk_process(F2_init, steps, 100, F2_range[0], F2_range[-1])
    F3_walk = random_walk_process(F3_init, steps, 100, F3_range[0], F3_range[-1])
    alpha2_walk = random_walk_process(alpha2_init, steps, 0.02, alpha2_range[0], alpha2_range[-1])
    alpha3_walk = random_walk_process(alpha3_init, steps, 0.02, alpha3_range[0], alpha3_range[-1])

    # Convert the random walks to a (999424,5) vector
    u = np.column_stack((F1_walk,F2_walk,alpha2_walk,F3_walk,alpha3_walk))

    # Return the u vector
    return u

# Number of steps in the random walk
samples = 999424    

# Generate you. 
u = apply_random_walk(F1, F2, F3, alpha2, alpha3, samples)

def plot_samples(samples):
    """
    Plot the random walks 
    """
    # Set up the plot
    plt.figure(figsize=(10, 8))
    plt.suptitle('Generated u Data via a Random Walk')
    
    # Plot F1
    plt.subplot(3, 2, 1)
    plt.plot(samples[0], label='F1')
    plt.title('Random Walk for F1')
    plt.xlabel('Time')
    plt.ylabel('F1 value')
    plt.grid(True)
    
    # Plot F2
    plt.subplot(3, 2, 2)
    plt.plot(samples[1], label='F2', color='orange')
    plt.title('Random Walk for F2')
    plt.xlabel('Time')
    plt.ylabel('F2 value')
    plt.grid(True)
    
    # Plot F3
    plt.subplot(3, 2, 3)
    plt.plot(samples[3], label='F3', color='green')
    plt.title('Random Walk for F3')
    plt.xlabel('Time')
    plt.ylabel('F3 value')
    plt.grid(True)
    
    # Plot alpha2
    plt.subplot(3, 2, 4)
    plt.plot(samples[2], label='alpha2', color='red')
    plt.title('Random Walk for alpha2')
    plt.xlabel('Time')
    plt.ylabel('alpha2 value')
    plt.grid(True)
    
    # Plot alpha3
    plt.subplot(3, 2, 5)
    plt.plot(samples[4], label='alpha3', color='purple')
    plt.title('Random Walk for alpha3')
    plt.xlabel('Time')
    plt.ylabel('alpha3 value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot the u vector
plot_samples(u.T)

def generate_tau(u):
    """
    Function to create the tau vector 
    """
    # Lengths of the boat
    l1 = -14
    l2 = 14.5
    l3 = -2.7
    l4 = 2.7

    # Preallocate an empty 3x999424 vector
    tau = np.zeros((3,u.shape[1]))
    
    # Loop through each u vector and compute its corresponding tau vector
    for idx in range(len(u[0])):
        # Extract current u index
        ui = u[:,idx]
        
        # Extract the uf 
        uf = np.array([ui[0],ui[1],ui[3]])

        # Extract the alphas
        alpha = np.array([ui[2],ui[4]])

        # Create the B matrix
        B = np.array([[0, np.cos(alpha[0]), np.cos(alpha[1])],
                      [1, np.sin(alpha[0]), np.sin(alpha[1])],
                      [l2, l1*np.sin(alpha[0]) - l3*np.cos(alpha[0]), l1*np.sin(alpha[1])-l4*np.cos(alpha[1])]])
        
        # Compute the tau and add it to the appropriate index
        tau[:,idx] = B@uf
    
    # Return the transpose of the tau matrix (999424, 3)
    return tau.T

# Generate Tau
tau_desired = generate_tau(u.T)


def plot_training_data(data,title):
    """
    Plot the tau
    """
    plt.figure(figsize=(10, 8))
    plt.suptitle(title)
    
    # Plot T1
    plt.subplot(3, 1, 1)
    plt.plot(data[:,0], label='Tau_surge')
    plt.title('Tau surge')
    plt.xlabel('Time')
    plt.ylabel('Tau surge')
    plt.grid(True)
    
    # Plot T2
    plt.subplot(3, 1, 2)
    plt.plot(data[:,1], label='Tau_sway', color='orange')
    plt.title('Tau_sway')
    plt.xlabel('Time')
    plt.ylabel('Tau_sway')
    plt.grid(True)
    
    # Plot T3
    plt.subplot(3, 1, 3)
    plt.plot(data[:,2], label='Tau_yaw', color='green')
    plt.title('Tau_yaw')
    plt.xlabel('Time')
    plt.ylabel('Tau_yaw')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot the tau vector
plot_training_data(tau_desired,'Generated tau dataset')

# Normalize the tau vector to have 0 mean and 1 std
tau_1_mean = np.mean(tau_desired[:,0])
tau_1_std = np.std(tau_desired[:,0])
tau_desired[:,0] = (tau_desired[:,0]-tau_1_mean)/tau_1_std

tau_2_mean = np.mean(tau_desired[:,1])
tau_2_std = np.std(tau_desired[:,1])
tau_desired[:,1] = (tau_desired[:,1]-tau_2_mean)/tau_2_std

tau_3_mean = np.mean(tau_desired[:,2])
tau_3_std = np.std(tau_desired[:,2])
tau_desired[:,2] = (tau_desired[:,2]-tau_3_mean)/tau_3_std

# Plot the normalized tau vector
plot_training_data(tau_desired, 'Normalized tau dataset')

# Define number of samples in a mini batch
batch_size = 1024  

# Create a list from 0 to 999424 in increments of 1024
batch_start = np.arange(0,tau_desired.shape[0],1024) # tau.shape[0] = 999424

# Preallocate an empty matrix of the desired size
tau_batch = np.zeros((976,1024,3))

# Loop through the batch list 
for batch_idx, tau_index in enumerate(batch_start):
    # Extract the 1024 sample
    splice = tau_desired[tau_index:tau_index+batch_size,:]
    # Add the sample to the correct index
    tau_batch[batch_idx,:,:] = splice

# Plot a batch of the normalized tau data
plot_training_data(tau_batch[0], 'A batch (1024 samples) of the normalized tau dataset')

# Create the F_max vector for the L2 loss function
F_max = torch.Tensor([30000.0,60000.0,3.14,60000,3.14]).to(device)

# Create the delta F_max vector for the L3 loss function 
delta_F_max = torch.Tensor([1000,1000,0.17,1000,0.17]).to(device)

# Create the ac top and bottom vector for the L4 loss function
ac_bot = torch.Tensor([-1.745, -1.396]).to(device)
ac_top = torch.Tensor([1.396, 1.745]).to(device)

# Create a training and test set from the 976 batches
training, test = train_test_split(tau_batch,train_size=0.8)

# Move the training and test datasets to the device
training = torch.tensor(training, dtype=torch.float32).to(device)
test = torch.tensor(test, dtype=torch.float32).to(device)

# Plot a batch of the training data
plot_training_data(training[0], 'A batch (1024 samples) of the training dataset' )


class AutoEncoder(torch.nn.Module):
    """
    Class of the AutoEncoder model
    """

    def __init__(self, input_dim=3, thruster_output_dim=5):
        """
        Object constructor
        """
        super(AutoEncoder, self).__init__()
        
        # Define the encoder sequential
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 4),  
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4, 5), 
            torch.nn.LeakyReLU(),
            torch.nn.Linear(5, thruster_output_dim), 
        )

        # Define the decoder sequential
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(thruster_output_dim,5), 
            torch.nn.LeakyReLU(),
            torch.nn.Linear(5, 4), 
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4, input_dim), 
        )
        
        
    def forward(self, tau_desired):
        """
        Define the forward function for the model
        """
        # Pass the tau desired to the model to compute the u_hat
        thruster_commands = self.encoder(tau_desired)

        # Pass the u_hat to be decoder to compute tau_hat
        tau_pred = self.decoder(thruster_commands)
        
        # Return both outputs 
        return tau_pred, thruster_commands


class ConstrainedControlLoss(torch.nn.Module):
    """
    Class to compute the custom loss 
    """

    def __init__(self, F_max, delta_F_max, alpha_bot,alpha_top, alpha_weights=[10e0,10e0,10e-1,10e-7,10e-7,10e-1] ):
        """
        Object constructor
        """
        super(ConstrainedControlLoss, self).__init__()
        self.F_max = F_max
        self.delta_F_max = delta_F_max
        self.alpha_weights = alpha_weights
        self.alpha_bot =alpha_bot 
        self.alpha_top =alpha_top

    def compute_tau_command(self,u_hat):
        """
        Function to compute tau command from the u_hat encoder output
        """
        # Define lengths of the boat
        l1 = -14
        l2 = 14.5
        l3 = -2.7
        l4 = 2.7
        
        # Define each variable of the B matrix in batch form
        B11 = torch.zeros((1024))
        B12 = torch.cos(u_hat[:,[2,4]][:,0])
        B13 = torch.cos(u_hat[:,[2,4]][:,1])
        B21 = torch.ones((1024))
        B22 = torch.sin(u_hat[:,[2,4]][:,0])
        B23 = torch.sin(u_hat[:,[2,4]][:,1])
        B31 = l2*torch.ones((1024))
        B32 = l1*torch.sin(u_hat[:,[2,4]][:,0])-l3*torch.cos(u_hat[:,[2,4]][:,0])
        B33 = l1*torch.sin(u_hat[:,[2,4]][:,1])-l4*torch.cos(u_hat[:,[2,4]][:,1])

        # Define the B matrix in batch form
        B_matrices = torch.stack([torch.stack([B11, B12, B13], dim=-1),
                                  torch.stack([B21, B22, B23], dim=-1),
                                  torch.stack([B31, B32, B33], dim=-1)], dim=1)
        
        # Compute tau command via batch matrix multiplication
        # Unsqueeze the u_hat to create the 3x1 vector 
        tau_command = torch.bmm(B_matrices,torch.unsqueeze(u_hat[:,[0,1,3]],dim=2))
        
        # Return the tau_command vector and squeeze to revert the vector to a 1024x3 versus a 1024x3x1
        return torch.squeeze(tau_command)
    
    
    def forward(self, tau_desired, tau_pred, u_hat):
        """
        Define the forward function for the loss function
        """
        # Get tau command
        tau_command = self.compute_tau_command(u_hat)

        # L0: MSE loss between tau and tau command
        L_ground = torch.nn.MSELoss()(tau_command, tau_desired) 

        # L1: MSE loss between tau and tau_hat 
        L_force = torch.nn.MSELoss()(tau_pred, tau_desired)
        
        # L2: Thruster command magnitude loss
        L_magnitude = torch.sum(torch.clamp(torch.abs(u_hat) - self.F_max, min=0))
        
        # L3: Rate changes loss 
        # Shift u_hat by 1 in the batch dimension
        u_hat_shifted = torch.roll(u_hat, shifts=-1, dims=0)
        # Compute absolute differences between u_hat and its shifted version
        rate_change = torch.abs(u_hat - u_hat_shifted)
        # Apply the rate change penalty: max(rate_change - delta_u_max, 0)
        rate_change_penalty = torch.nn.functional.relu(rate_change - self.delta_F_max)
        # Sum the penalty over the batch (1024) and variable dimensions (5)
        L_rate = rate_change_penalty.sum()

        # L4: Power consumption minimization loss
        L_power = torch.sum(abs(u_hat[:,[0,1,3]])**(3/2))

        # L5: Azimuth sector loss
        L_5_bot = torch.sum((u_hat[:,[2,4]] < self.alpha_bot[1]) * (u_hat[:,[2,4]] > self.alpha_bot[0]))
        L_5_top = torch.sum((u_hat[:,[2,4]] < self.alpha_top[1]) * (u_hat[:,[2,4]] > self.alpha_top[0]))
        L_azimuth = torch.add(L_5_bot,L_5_top)

        # Combined loss
        total_loss = (self.alpha_weights[0] * L_ground
                      + self.alpha_weights[1] * L_force 
                      + self.alpha_weights[2] * L_magnitude
                      + self.alpha_weights[3] * L_rate
                      + self.alpha_weights[4] * L_power
                      + self.alpha_weights[5] * L_azimuth 
                      )
        
        # Append losses for plotting purposes
        individual_losses = [self.alpha_weights[0] * L_ground.detach().cpu().numpy(),
                             self.alpha_weights[1] * L_force.detach().cpu().numpy(),
                             self.alpha_weights[2] * L_magnitude.detach().cpu().numpy(),
                             self.alpha_weights[3] * L_rate.detach().cpu().numpy(),
                             self.alpha_weights[4] * L_power.detach().cpu().numpy(),
                             self.alpha_weights[5] * L_azimuth.detach().cpu().numpy()]
        
        # Return all losses
        return total_loss, tau_command, individual_losses

# Create a loss function object
loss_function = ConstrainedControlLoss(F_max, delta_F_max, ac_bot,ac_top)

# Create a model object
model = AutoEncoder().to(device) 

# Create an optimizer function with L2 normalization
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Define number of epochs
num_epochs = 200

# Define lists for data collection
tau_des_data            = []
loss_data               = []
tau_pred_data           = []
tau_cmd_data            = []
u_hat_data              = []
loss_per_epoch          = []
loss_per_batch          = []
test_loss_per_batch     = []
test_loss_per_epoch     = []
individual_losses       = []
individual_losses_test  = []

# Create a list contains the index order from 0 to 780. This will be used to index the batches for training.
# The shuffle occurs at the end of each epoch
shuffled_training = np.arange(0, training.shape[0])

# Main training and evaluation loop. Runs as many times as number of epochs.
for epoch in range(num_epochs):
    # Set the model to train
    model.train()

    # Loop through the shuffled training index list
    for index in shuffled_training:
        # Extract the index to train
        tau_des = training[index,:,:] 

        # Append the data for plotting
        tau_des_data.append(tau_des.detach().cpu().numpy())

        # Pass the batch to the model 
        tau_pred, u_hat  = model(tau_des)

        # Append the model output data for plotting
        tau_pred_data.append(tau_pred.detach().cpu().numpy().T)
        u_hat_data.append(u_hat.detach().cpu().numpy().T)

        # Pass the model outputs to the loss function to compute the loss
        loss, tau_command, individual_loss_values = loss_function(tau_des, tau_pred, u_hat)

        # Append the loss outputs for plotting
        tau_cmd_data.append(tau_command.detach().cpu().numpy().T)
        individual_losses.append(individual_loss_values)

        # Step the optimizier
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Append the loss for the batch 
        loss_per_batch.append(loss.item())

    # Append the loss for the epoch
    loss_per_epoch.append(loss_per_batch[-1])

    # Set the model to evaluation mode
    model.eval() 
    # Set to no optimization
    with torch.no_grad(): 
        # Loop through the test data
        for batch_idx, batch in enumerate(test):
            # Pass the batch to the model
            tau_pred, u_hat = model(batch) 
            
            # Pass the model outputs to the loss function to compute the loss 
            loss, tau_command, individual_loss_values = loss_function(batch, tau_pred, u_hat)

            # Append the loss outputs for plotting
            individual_losses_test.append(individual_loss_values)
            
            # Append the loss for the batch 
            test_loss_per_batch.append(loss.item())
    
    # Append the loss for the epoch
    test_loss_per_epoch.append(test_loss_per_batch[-1])
    
    # Print updates to the command line
    print(f"Epoch: {epoch+1}\t | Training Loss: {loss_per_epoch[-1]}\t | Testing Loss: {test_loss_per_epoch[-1]}")

    # Shuffle the training dataset
    shuffled_training = shuffle(shuffled_training)

# Plot the various training losses
plt.figure(figsize=(10, 8))
plt.plot(loss_per_batch)
plt.title('Training Loss per sample')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(loss_per_epoch)
plt.title('Training Loss per epoch')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(test_loss_per_batch)
plt.title('Testing Loss per sample')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(test_loss_per_epoch)
plt.title('Testing Loss per epoch')
plt.show()

def plot_individual_losses(samples, title):
    """
    Plot the individual training losses
    """

    # Set up the plot
    plt.figure(figsize=(10, 8))
    plt.suptitle(title)
    
    # Plot L0
    plt.subplot(3, 2, 1)
    plt.plot(samples[:,0], label='L0')
    plt.title('L0 Loss')
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot L1
    plt.subplot(3, 2, 2)
    plt.plot(samples[:,1], label='L1', color='orange')
    plt.title('L1 Loss')
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot L2
    plt.subplot(3, 2, 3)
    plt.plot(samples[:,2], label='L2', color='green')
    plt.title('L2 Loss')
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot L3
    plt.subplot(3, 2, 4)
    plt.plot(samples[:,3], label='L3', color='red')
    plt.title('L3 Loss')
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot L4
    plt.subplot(3, 2, 5)
    plt.plot(samples[:,4], label='L4', color='purple')
    plt.title('L4 Loss')
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.grid(True)

    # Plot L5
    plt.subplot(3, 2, 6)
    plt.plot(samples[:,5], label='L5', color='purple')
    plt.title('L5 Loss')
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot the individual training losses
plot_individual_losses(np.asarray(individual_losses),'Individual Losses during Training')

# Plot the individual training losses
plot_individual_losses(np.asarray(individual_losses_test), 'Individual Losses during Testing')

def plot_comparsion_data(data, pred, toggle_axis,title):
    """
    Plot side by side tau vs tau cmd or hat data
    """
    plt.figure(figsize=(10, 8))
    plt.suptitle(title)
    
    # Plot T1
    plt.subplot(3, 2, 1)
    plt.plot(data[0], label='Tau_surge')
    plt.title('Tau surge')
    plt.xlabel('Time')
    plt.ylabel('Tau surge')
    plt.grid(True)

    surge_x = plt.gca().get_xlim()
    surge_y = plt.gca().get_ylim()
    
    # Plot T2
    plt.subplot(3, 2, 3)
    plt.plot(data[1], label='Tau_sway', color='orange')
    plt.title('Tau_sway')
    plt.xlabel('Time')
    plt.ylabel('Tau_sway')
    plt.grid(True)

    sway_x = plt.gca().get_xlim()
    sway_y = plt.gca().get_ylim()
    
    # Plot T3
    plt.subplot(3, 2, 5)
    plt.plot(data[2], label='Tau_yaw', color='green')
    plt.title('Tau_yaw')
    plt.xlabel('Time')
    plt.ylabel('Tau_yaw')
    plt.grid(True)

    yaw_x = plt.gca().get_xlim()
    yaw_y = plt.gca().get_ylim()

    # Plot T1 prediction
    plt.subplot(3, 2, 2)
    plt.plot(pred[0], label='Tau_surge')
    plt.title('Predicted Tau surge')
    plt.xlabel('Time')
    plt.ylabel('Tau surge')
    plt.grid(True)

    if toggle_axis:
        plt.gca().set_xlim(surge_x)
        plt.gca().set_ylim(surge_y)

    # Plot T2 prediction
    plt.subplot(3, 2, 4)
    plt.plot(pred[1], label='Tau_sway', color='orange')
    plt.title('Predicted Tau_sway')
    plt.xlabel('Time')
    plt.ylabel('Tau_sway')
    plt.grid(True)

    if toggle_axis:
        plt.gca().set_xlim(sway_x)
        plt.gca().set_ylim(sway_y)
    
    # Plot T3 prediction
    plt.subplot(3, 2, 6)
    plt.plot(pred[2], label='Tau_yaw', color='green')
    plt.title('Predicted Tau_yaw')
    plt.xlabel('Time')
    plt.ylabel('Tau_yaw')
    plt.grid(True)

    if toggle_axis:
        plt.gca().set_xlim(yaw_x)
        plt.gca().set_ylim(yaw_y)
    
    plt.tight_layout()
    plt.show()

# Plot the comparison between the tau and tau hat for training
plot_comparsion_data(np.asarray(tau_des_data)[-1,:,:].T,np.asarray(tau_pred_data)[-1,:,:],True, 'Comparison between input data and decoder output')

print(f'Training: Final loss in the last batch when comparing the y vs yhat: {torch.nn.MSELoss()(torch.from_numpy(np.asarray(tau_pred_data)[-1,:,:]),torch.from_numpy(np.asarray(tau_des_data)[-1,:,:].T))}')

# Plot the comparison between the tau and tau cmd for training
plot_comparsion_data(np.asarray(tau_des_data)[-1,:,:].T,np.asarray(tau_cmd_data)[-1,:,:],True, 'Comparison between input data and generated tau via encoder output')

print(f'Training: Final loss in the last batch when comparing the y vs ycmd: {torch.nn.MSELoss()(torch.from_numpy(np.asarray(tau_cmd_data)[-1,:,:]),torch.from_numpy(np.asarray(tau_des_data)[-1,:,:].T))}')

# Run one batch through the model for plotting
model.eval() 
with torch.no_grad():
    test_output, test_u = model(test[0])

    test_loss, test_tau_command, individual_loss_values = loss_function(test[0], test_output, test_u)

# Plot the comparison between the tau and tau hat for testing
plot_comparsion_data(np.asarray(test)[0,:,:].T,np.asarray(test_output).T, True,  'Comparison between test input data and decoder output')

print(f'Testing: Final loss in the last batch when comparing the y vs yhat: {torch.nn.MSELoss()(torch.from_numpy(np.asarray(test_output).T),torch.from_numpy(np.asarray(test)[0,:,:].T))}')

# Plot the comparison between the tau and tau cmd for testing
plot_comparsion_data(np.asarray(test)[0,:,:].T,np.asarray(test_tau_command).T,True,'Comparison between test input data and generated tau via encoder output')

print(f'Testing: Final loss in the last batch when comparing the y vs ycmd: {torch.nn.MSELoss()(torch.from_numpy(np.asarray(test_tau_command).T),torch.from_numpy(np.asarray(test)[0,:,:].T))}')

