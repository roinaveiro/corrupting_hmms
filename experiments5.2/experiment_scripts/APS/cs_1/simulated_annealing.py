import numpy as np
from scipy.special import softmax


def return_mat(x, z, i):
    z[i] = x
    return z

def s(t, den=5, l=5):
    return np.exp(-l*t/den) # Watch out!
    
def generate_candidate(t, z_init, attacker, N):

    idx = np.random.choice(attacker.T)
    Z_candidates = np.apply_along_axis( 
        lambda x: return_mat(x, z_init, idx), 1, np.eye(attacker.n_obs))

    energies = np.zeros( len(Z_candidates) )

    for i, z in enumerate(Z_candidates):
        energies[i] = attacker.expected_utility(z, N)/s(t, den=attacker.T)


    p = softmax(energies)
    candidate_idx = np.random.choice( 
        np.arange(len(Z_candidates) ), p=p )
    
    return Z_candidates[candidate_idx]


def simulated_annealing(attacker, n_iter, N=10, verbose=True):

    # Z_set = attacker.generate_attacks()
    # z_init = Z_set[ np.random.choice(Z_set.shape[0]) ]
    z_init = attacker.sample_attack()

    for t in np.arange(n_iter):

        if verbose:
            if t%50 == 0:
                print("Percentage completed:", 
                np.round( 100*t/n_iter, 2) )
                print("Current state", z_init)

        z_init = generate_candidate(t, z_init, attacker, N)
    

    return z_init




if __name__ == "__main__":

    pass

