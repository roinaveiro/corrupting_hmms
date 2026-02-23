import numpy as np
from joblib import Parallel, delayed




def MC_enumeration(attacker, N=10, verbose=True):
    
    Z_set = attacker.generate_attacks()
    utilities = np.zeros(Z_set.shape[0])

    for i, z in enumerate(Z_set):

        if verbose:
            if i%1000 == 0:
                print("Percentage completed:", 
                np.round(100*i/len(utilities), 2)  )
                
        utilities[i] = attacker.expected_utility(z, N=N)

    z_star = Z_set[np.argmax(utilities)]
    
    return z_star, utilities


def MC_enumeration_parallel(attacker, N=10, verbose=True, num_jobs = -2):
    
    Z_set = attacker.generate_attacks()

    utilities = np.array(Parallel(n_jobs=num_jobs)(delayed(attacker.expected_utility)(z,N) for z in (Z_set)))

    z_star = Z_set[np.argmax(utilities)]
    
    return z_star, utilities





if __name__ == "__main__":
    pass

