Code generates a neural network 
			f: R x R^4 --> R^4 
to approximate the flow map of the Hamiltonian vector field of the Hamiltonian
H(q_1,q_2,p_1,p_2) = 1/2*(p_1^2 + p_2^2 + q_1^2 + q_2^2) + lam*(q_1^2 + q_2^2)^2
where lam is a constant. The diagonal SO(2) action on (q,p) gives the Lie group symmetries
of the flow. Solutions to this Hamiltonian have no closed form solutions, so we approximate
the flow using the symplectic leapfrog method.

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