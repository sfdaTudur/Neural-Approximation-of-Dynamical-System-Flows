Consider the system q'' = -q, where q(t) \in R^3.

Writing v=q', the first order system is q'= v, v' = -q.
The state is z = (q,v) \in R^6.
The flow map of this system is

\Psi(t,q,v) = (qcos(t) + vsin(t), -qsin(t) + vcos(t))

which is invariant under the diagonal action of SO(3).

We train a neural network to predict the flow map above, and
test empirical equivariance on out of distribution data.