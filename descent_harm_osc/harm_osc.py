import torch
import sys
import math
import csv
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


#Use GPU if device has one, else use CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


#ReLU neural net with specified depth, used to approximate the flow of a dynamical system
class NeuralNetwork(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth

        self.first_layer = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU()
        )

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(8, 8),
                nn.ReLU()
            )
            for _ in range(depth)
        ])

        self.final_layer = nn.Linear(8, 2)

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.final_layer(x)


#---------------------Training data for the flow of the Harmonic oscillator-------------

def psi_flow(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Ψ(t, (a,b)) = (a cos t - b sin t, a sin t + b cos t)
    flow of the harmonic oscillator z' = Jz with J the matrix
    representing multiplication by i as an R-linear map.

    t: shape (N,) or (N,1)
    x: shape (N,2)
    so we have N samples with t[i] the time associated with x[i] = (a_i,b_i)
    returns: shape (N,2)
    """
    t = t.view(-1, 1)  #reshapes t into a column vector of shape (N,1)
    a = x[:, 0:1]  #take all rows and column 0, gives first coordinate of each batch
    b = x[:, 1:2]  #take all rows and column 1, gives second coordinate of each batch
    cos_t = torch.cos(t)  #apply cos elementwise to each t_i
    sin_t = torch.sin(t)
    y1 = a * cos_t - b * sin_t
    y2 = a * sin_t + b * cos_t
    return torch.cat([y1, y2], dim=1)

#generate random timesteps and random points in the plane

def make_training_data(batch_size, R):
    #t ~ Uniform[0,2\pi]
    T = 2 * math.pi *torch.rand(batch_size,device = device, dtype=torch.float32)
    #(x,y)~Uniform(D(0,R)), to sample uniformly in area, need radius r
    #of random variable to be r~ R\sqrt(u) for u~Uniform(0,1)
    u = torch.rand(batch_size, device=device, dtype=torch.float32)
    r = R * torch.sqrt(u)
    theta = 2 * math.pi * torch.rand(batch_size, device=device, dtype=torch.float32)
    x = r* torch.cos(theta)
    y = r* torch.sin(theta)
    pts = torch.cat((x.view(-1,1), y.view(-1,1)), dim=1)
    inputs = torch.cat((T.view(-1, 1), x.view(-1,1), y.view(-1,1)), dim=1)
    return inputs, psi_flow(T, pts)



"""
# Testing
t = torch.randn(3)  #randn samples from the normal distribution, while torch.rand() samples from the uniform distribution
x = torch.randn(3)
y = torch.randn(3)
print(t)
print(x)
print(y)
#print(torch.stack([t.view(-1,1),x.view(-1,1), y.view(-1,1)], dim=1))   #gives batch of column vectors..
data = torch.cat((t.view(-1,1),x.view(-1,1),y.view(-1,1)), dim=1)   #makes a matrix with rows (t_i,x_i,y_i)
print(data)
print(data[:, 0:1])  #returns column 0 as a column vector (all t_i's in batch)
print(data[:, 1:2])  #column 1
print(data[:, 2:3])  #column 2
print(data[:, 1:2] * torch.cos(data[:, 0:1]))  #test componentwise mult
print(make_training_data(3,3))
"""





#--------------------------------Training--------------------------------------------
def train_model(model, depth, criterion, optimizer, dl, radius, batch_size, no_of_epochs):
    model.train()
    for epoch in range(1, no_of_epochs + 1):
        running_loss = 0.0
        for batch_inputs, batch_targets in dl:
            optimizer.zero_grad()
            preds = model(batch_inputs)
            loss = criterion(preds, batch_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_inputs.size(0)
        epoch_loss = running_loss / len(dl.dataset)
        print(f"Epoch {epoch:03d} | MSE: {epoch_loss:.6f}")

    # Save weights
    save_path = f"data/model_weights_depth_{depth}.pth"

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "depth": depth,
            "radius": radius,
            "batch_size": batch_size,
            "epochs": no_of_epochs,
        },
        save_path,
    )

    print(f"Saved model weights to {save_path}")


    

#---------------------------------Set parameters for training------------------------------
max_depth = 50   #maximum depth of nn architecture to be used as our model
dataset_size = 60000   #size of training set 
radius = 5    #radius of sampled (x,y)
batch_size = 1024   #size of one batch for gradient descent
no_of_epochs = 5   #epochs for training
inputs, targets = make_training_data(dataset_size, radius)
ds = torch.utils.data.TensorDataset(inputs, targets)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)


for depth in range(1,max_depth):   #train models of varying depth
    model = NeuralNetwork(depth).to(device)
    criterion = nn.MSELoss()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)   # Stochastic Gradient Descent
    train_model(model, depth, criterion, optimizer, dl, radius, batch_size, no_of_epochs)
    




#-----------------------------Equivariance test--------------------------------


def rot_matrix(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    G = torch.stack(
        [
            torch.stack([c, -s], dim=1),
            torch.stack([s,  c], dim=1),
        ],
        dim=1,
    )
    return G



"""
#test function above 
theta = torch.rand(4)
x = torch.rand(3)
y = torch.rand(3)
data = torch.cat((x.view(-1,1),y.view(-1,1)), dim=1)
data_col = data.unsqueeze(-1)
G = rot_matrix(theta)
print(theta)
print(G)
print(data)
print(data_col)
print(apply_rot(G, data_col))
"""



def empirical_equivariance(model, num_grp_elts, num_datapts, R):
    model.eval()
    error_sum = 0

    with torch.no_grad():
        for i in range(0,num_grp_elts):
            theta = 2* math.pi * torch.rand(1, device = device, dtype=torch.float32)
            G = rot_matrix(theta)
            for j in range(0,num_datapts):
                t = 2 * math.pi *torch.rand(1,device = device, dtype=torch.float32)
                u = torch.rand(1, device=device, dtype=torch.float32)
                r = R * torch.sqrt(u)
                phi = 2 * math.pi * torch.rand(1, device=device, dtype=torch.float32)
                x = r* torch.cos(phi)
                y = r* torch.sin(phi)
                v = torch.cat((x,y), dim = 0).view(-1,1)
                Gv = G@v
                Gv_row = Gv.squeeze(-1)
                t_row = t.view(1,1)
                t_Gv = torch.cat((t_row, Gv_row), dim=1).to(device)
                f_tGv = model(t_Gv)
                v_row = v.T
                f_t_v = model(torch.cat((t_row, v_row), dim=1).to(device))  #output is a row
                f_t_v_col = f_t_v.unsqueeze(-1)   #turn into column
                G_f_t_v = (G @ f_t_v_col).squeeze(-1)
                error = torch.norm(f_tGv - G_f_t_v)
                error_sum += error.item()
    return error_sum / (num_grp_elts * num_datapts)



"""
# testing for equivariance part of code
x = torch.rand(1)
y = torch.rand(1)
print(x)
print(y)
print(torch.cat((x,y), dim=0))
print(torch.cat((x,y), dim=0).view(-1,1))
vect = torch.cat((x,y), dim=0).view(-1,1)
theta = theta = 2* math.pi * torch.rand(1, device = device, dtype=torch.float32)
G = rot_matrix(theta)
print(theta)
print(G)
Gv = G@vect
theta_row = theta.view(1,1)
print(Gv)
print(Gv.squeeze(-1))    #put Gv into row form for inputting into nn
print(torch.cat((theta_row, Gv.squeeze(-1)), dim=1))
"""



#--------------Store (depth,equivariance) table to study double-descent-----------------------
num_grp_elt = 100
num_datapt = 100
R = 20    #The random inputs sampled from a disk larger radius than used for training


with open("data/empirical_equivariance_vs_depth.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["depth", "eq_error"])  # header

    for depth in range(1, max_depth):
        model = NeuralNetwork(depth).to(device)

        checkpoint = torch.load(
            f"data/model_weights_depth_{depth}.pth",
            map_location=device
        )
        model.load_state_dict(checkpoint["state_dict"])

        eq_error = empirical_equivariance(model, num_grp_elt, num_datapt, R)

        writer.writerow([depth, eq_error])