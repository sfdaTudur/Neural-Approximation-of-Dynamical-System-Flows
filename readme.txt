Code generates a neural network 
			f: R x R^2 --> R^2 
to approximate the flow map of the Harmonic oscillator z' = Jz,
where z:R --> R^2 is a curve, and J is the matrix [[0,-1],[1,0]].

For a trained neural net approximating the flow, we test for SO(2)-equivariance to see if the neural
net is learning the Lie point symmetries for the system. To do this, let g be rotation by an irrational 
multiple of pi. We then look at the average error
		1/N \sum_{i=1}^N ||f(t,g^ix) - g^i f(t,x)||^2
for chosen values of (t,x).
We also plot the points f(t,g^ix) for i=1,..,N, to see whether the topological 
type of the SO(2) orbit is preserved under the neural net.

To run the code:
1-- Run data_gen.py after modifying the --out argument in the main() function to set the file location to save training data.
2-- Run train.py after modifying the --data and --save arguments in main() to point the dataset path and the desired checkpoint path.
3-- Run equivariance_test.py after setting the --weights argument in main() to the location of saved weights.

Files:
model.py -- Defines the two layer ReLU network used to approximate flow
data_gen.py -- Generates the training dataset for the harmonic oscillator flow.
train.py -- Builds and trains the neural net using mean squared error loss.
equivariance_test.py  -- tests SO(2)-equivariance to gauge generalization