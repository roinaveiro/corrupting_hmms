import numpy as np
from scipy.special import softmax


def return_mat(x, z, i):
    z[i] = x
    return z

def s(t, l=1):
    return np.exp(-l*t/5)
    
def generate_candidate(t, z_init, attacker):

    idx = np.random.choice(5)
    Z_candidates = np.apply_along_axis( 
        lambda x: return_mat(x, z_init, idx), 1, np.eye(6))

    energies = np.zeros( len(Z_candidates) )

    for i, z in enumerate(Z_candidates):
        energies[i] = attacker.expected_utility(z, N=10)/s(t)


    p = softmax(energies)
    candidate_idx = np.random.choice( 
        np.arange(len(Z_candidates) ), p=p )
    
    return Z_candidates[candidate_idx]


def simulated_annealing(attacker, n_iter, verbose=True):

    Z_set = attacker.generate_attacks()
    z_init = Z_set[ np.random.choice(Z_set.shape[0]) ]

    for t in np.arange(n_iter):

        if verbose:
            if t%50 == 0:
                print("Percentage completed:", 
                np.round( 100*t/n_iter, 2) )
                print("Current state", z_init)

        z_init = generate_candidate(t, z_init, attacker)
    

    return z_init




if __name__ == "__main__":

    pass

