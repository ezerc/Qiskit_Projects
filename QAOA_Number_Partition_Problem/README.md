# NUMBER PARTITION PROBLEM QAOA

The number partition problem consists of partitioning a given set S into two subsets S1 and S2, such that the difference between the sum of the elements in each partitioned set is minimal. This problem is known to be NP-hard, and can be approached as an optimization problem in which the difference between sums is the variable to be minimized. In this project, we approach this problem by resorting to the Quantum Approximation Optimization Algorithm (QAOA) [1].

## CONTENTS

The project includes two documents. First, a main Python script containing a function to create the QAOA Hamiltonian associated with the number partition problem and a class used to solve this problem using the QAOA. Second, a Jupyter Notebook with a few test examples and a brief description of how to solve the problem using the main document.

## INSTALLING

Qiskit needs to be installed in your local environment. This can be done using pip:
> pip install qiskit

In order to use the QPU option you need an IBM experience account, which can be easily created here: https://quantum-computing.ibm.com/. This will grant you with a personal token id which you then need to activate to use the QPU. This can be done by running the commands:

## GETTING STARTED

The Jupyter Notebook accompanying this project contains several tests using different sets and describing how to use the number_partition_qaoa() class. The workflow can be described as follows:

- Define a set S to be partitioned as a list:
> S = [...]

- Feed the set to the fully_connected_model(S) function to extract the Hamiltonian variables of the set, a list of graph edges and the interaction matrix:
> E,J = fully_connected_model(S)

- Feed the variables to the class to create a new instance:
> np_test = number_partition_qaoa(S,E,J)

- Use the run_qaoa() method to solve the number partition problem using QAOA. The method can take several inputs: how many layers to consider, the size of the grid search to initialise the optimization, the number of times to run the circuit, both during the grid search and during optimization, and whether we want to use a QPU to solve the final part of the problem.
> S, R, out = np_test.run_qaoa(p,num_samples,grid_num_samples,grid_size,use_qpu)

The run_qaoa() method returns two lists corresponding to each set partition and a dictionary containing the measurement outcomes of the circuit and their frequency. The outcomes can be neatly displayed in decreasing frequency order by calling the state_results(out) method. Finally, in order to benchmark our results for moderate set sizes, the states_and_energies() method can be used to retrieve a dictionary with all configurations and their associated energy.

## METHOD DETAILS

The number partition problem can be reformulated as finding the ground state of a quantum system, described by an all-to-all interacting Ising Hamiltonian [2]. The matrix elements of the Hamiltonian are given by the particular elements in the set.

The QAOA is an algorithm designed to find the ground state of a given Hamiltonian, result of a mapping from the cost function of the given optimization problem. This is done by creating a variational ansatz, with a set of parameters to be chosen such that this ansatz minimizes the expectation value of the Ising Hamiltonian. In particular, we use the sign convention of [1] to create the unitaries. The symmetries of the problem restrict the Ising Hamiltonian angles in a range from 0 to π and the mixing Hamiltonian angles from 0 to π/2.

The optimal variational parameters are found via classical optimization, using the Nelder-Mead algorithm. The values to initialize this optimisation process are obtained by performing a coarse grained grid search and keeping the parameters that yield the lowest expectation value [3]. This is performed by searching one pair of parameters at a time and fixing the pair to the best value before proceeding to the next, i.e. layer by layer, in the QAOA terminology.

Taking the measurement results from running the quantum circuit with optimal values, we select the five most frequent and use them to parti
tion the original set. The partition that yields the smallest difference of sums is chosen as the final answer.

## REFERENCES

[1] E. Farhi et al, A Quantum Approximate Optimization Algorithm, arXiv:1411.4028

[2] A. Lucas, Ising formulations of many NP problem, Front. Phys., 12 Feb. 2014

[3] A. H. Karamlou et al, Analyzing the Performance of Variational Quantum Factoring on a Superconducting Quantum Processor, arXiv:2012.07825
