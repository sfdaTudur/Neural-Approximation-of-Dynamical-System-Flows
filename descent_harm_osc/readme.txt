Code generates a neural network 
			f: R x R^2 --> R^2 
to approximate the flow map of the Harmonic oscillator z' = Jz,
where z:R --> R^2 is a curve, and J is the matrix [[0,-1],[1,0]].

For a trained neural net approximating the flow, we test for SO(2)-equivariance to see if the neural
net is learning the Lie point symmetries for the system.

Pick a distribution \mu over (t,x) and a Haar-uniform g \in SO(2).
let g_1,..,g_n be i.i.d. random elements of SO(2), and let (t_1,x_1),..,(t_m,x_m) be random
elements of R x R^2, then define the empirical equivariance estimator to be
        1/mn \sum_{i=1, j=1}^{n,m} ||f(t_j, g_i x_j) - g_i f(t_j,x_j)||^2.

We also plot the points f(t,g^ix) for i=1,..,N, to see whether the topological 
type of the SO(2) orbit is preserved under the neural net.

