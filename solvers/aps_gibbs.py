import numpy as np
from scipy.special import softmax
from solvers.simulated_annealing import return_mat

def update_probs_metropolis(H, attacker, hmms_old, value_old, z):
    
    hmms = []
    value = 0
    
    for h in range(H):
        
        hmm_sample = attacker.sample_hmm()
        value += np.log( attacker.utility(z, hmm_sample) )
        hmms.append(hmm_sample)
        
    value /= H
    prob =  np.exp(H*value - H*value_old)
    
    if np.random.uniform() < prob:
        
        #print("Accept!")
        return hmms, value
    
    else:
        
        return hmms_old, value_old
    
def update_z(attacker, hmms, z, idx):
    
    Z_candidates = np.apply_along_axis( 
        lambda x: return_mat(x, z, idx), 1, np.eye(attacker.n_obs))

    energies = np.zeros( len(Z_candidates) )

    for i, z_new in enumerate(Z_candidates):
        
        for hmm_sample in hmms:
            energies[i] += np.log( attacker.utility(z_new, hmm_sample) )
            
    p = softmax(energies)
    #print(p)

    
    candidate_idx = np.random.choice( 
        np.arange(len(Z_candidates) ), p=p )
    
    return Z_candidates[candidate_idx]


def get_mode(samples):
    vals, counts = np.unique( samples, return_counts=True, axis=0 )
    index = np.argmax(counts)
    return vals[index]

def aps_gibbs(cooling_schedule, attacker, burnin=0.2, verbose=True):
    
    Z_set = attacker.generate_attacks()
    z_init = Z_set[ np.random.choice(Z_set.shape[0]) ]
                   
    z_samples = np.zeros( [len(cooling_schedule),
                           z_init.shape[0], z_init.shape[1]] )
                   
    hh = cooling_schedule[0]

    ##
    hmms = []
    value = 0

    for temp in range(hh):

        hmm_sample = attacker.sample_hmm()
        value += np.log( attacker.utility(z_init, hmm_sample) )
        hmms.append(hmm_sample)


    value /= hh


    for i, temp in enumerate(cooling_schedule):
        
        if verbose:
            
            if i%10 == 0:
                print("Percentage completed:", 
                np.round( 100*i/len(cooling_schedule), 2) )
                print("Current state", z_init)
            

        # Update z
        for idx in range(attacker.T):
            z_init = update_z(attacker, hmms, z_init, idx)
            z_samples[i, idx] = z_init[idx]

        # Update value with new z
        value = 0
        for n in range(len(hmms)):
            value += np.log( 
                attacker.utility(z_init, hmms[n]) )
        value /= len(hmms)

        hmms, value = update_probs_metropolis(temp, 
                                              attacker, hmms, value, z_init)
        
    z_star = np.zeros_like(z_init)
    burnin_end = np.int(burnin * z_samples.shape[0])

    for i in range(z_star.shape[0]):
        z_star[i] = get_mode(z_samples[  burnin_end:, i ])
        
    return z_star, z_samples




if __name__ == "__main__":

    pass

