import numpy as np
import ndd
from math import factorial
import multiprocessing as mp

### Functions for performing IB analyses on empirical data ###

# compute mutual information using NSB method (use for actual human choice behavior)
def mutual_inf_nsb(x,y,ks):
    """
    Calculate mutual information using NSB method
    """
    ar = np.column_stack((x,y))
    mi = ndd.mutual_information(ar,ks)
    return np.log2(np.e)*mi #ndd returns nats - multiply by log2(e) to convert to bits

### Functions for computing IB bounds/curves ###

# compute I(X;R) using distributions, standalone version
def compute_I_XR(p_RgX,p_XgY,p_Y):
    p_X = np.sum(p_XgY * p_Y, axis=1, keepdims=True)
    p_R = np.sum(p_RgX * p_X, axis=0, keepdims=True)
    return np.sum(p_RgX*p_X*np.log2(p_RgX/p_R))

# compute I(X;R) using distributions, version for use within IB algorithm (where p_X and p_R are already computed and passed in as arguments)
# def compute_I_XR_(p_RgX,p_X,p_R):
#     return np.sum(p_RgX*p_X*np.log2(p_RgX/p_R))

# compute I(R;Y) using distributions, standalone version
def compute_I_RY(p_RgX,p_XgY,p_Y):
    p_XY = p_XgY * p_Y
    p_X = np.sum(p_XY, axis=1, keepdims=True)
    p_YgX = p_XY / p_X
    p_R = np.sum(p_RgX * p_X, axis=0, keepdims=True)
    p_XgR = (p_RgX * p_X) / p_R
    p_YgR =  p_YgX.T @ p_XgR
    return np.sum(p_YgR*p_R*np.log2(p_YgR/p_Y.reshape(-1,1)))

# compute I(R;Y) using distributions, version for use within IB algorithm (where p_YgR, p_R, and p_Y are already computed and passed in as arguments)
# def compute_I_RY_(p_YgR,p_R,p_Y):
#     return np.sum(p_YgR*p_R*np.log2(p_YgR/p_Y.reshape(-1,1)))

# compute I(X;R) using empirical list of choice probabilities for each trial (for model- or algorithm-computed choice probabilities only)
def compute_I_XR_emp(p_RgX_emp):
    p_RX_emp = p_RgX_emp * (1/N_emp)
    return np.sum(p_RX_emp*np.log2(p_RgX_emp/np.sum(p_RX_emp, axis=0, keepdims=True)))

# compute I(R;Y) using empirical list of choice probabilities and empirical list of Y for each trial (for model- or algorithm-computed choice probabilities only)
def compute_I_RY_emp(p_RgX_emp, Yemp):
    Yemp_oh = np.eye(N_Y)[Yemp.astype(int)]
    p_Y_emp = np.sum(Yemp_oh,axis=0,keepdims=True) / N_emp
    p_RX_emp = p_RgX_emp * (1/N_emp)
    p_R_emp = np.sum(p_RX_emp, axis=0, keepdims=True)
    p_XgR_emp = p_RX_emp / p_R_emp
    p_YgR_emp = Yemp_oh.T @ p_XgR_emp
    return np.sum(p_YgR_emp * p_R_emp * np.log2(p_YgR_emp/p_Y_emp.reshape(-1,1)))

# compute IB cost function value (standalone version)
def compute_IB_fn(beta,p_RgX,p_XgY,p_Y):
        Ixr = compute_I_XR(p_RgX,p_XgY,p_Y)
        Iry = compute_I_RY(p_RgX,p_XgY,p_Y)
        return Ixr - beta*Iry

# compute IB cost function value (for use within the IB algorithm)
# def compute_IB_fn_(beta,p_RgX,p_X,p_R,p_YgR,p_Y):
#         Ixr = compute_I_XR_(p_RgX,p_X,p_R)
#         Iry = compute_I_RY_(p_YgR,p_R,p_Y)
#         return Ixr - beta*Iry

# a function for attempting to sort the rows of p_YgR (for an IB solution) so that the maximum value in each row is on the diagonal
# this is for the purposes of comparing with the softmax solution
def get_map_inds(p_YgR,axis=1):
    p_YgR_copy = p_YgR.copy()
    if axis == 1:
        p_YgR_copy = p_YgR_copy.T
    mask = np.zeros_like(p_YgR_copy,dtype=bool)
    map_inds = []
    for ii, p_Ygr in enumerate(p_YgR_copy):
        p_Ygr_copy = p_Ygr.copy()
        p_Ygr_copy[map_inds] = -np.inf
        map_inds.append(np.argmax(p_Ygr_copy))
        mask[ii,np.argmax(p_Ygr_copy)] = True
    if axis == 1:
        mask = mask.T
    inv_map_inds = np.zeros_like(map_inds)
    for ii, ind in enumerate(map_inds):
        inv_map_inds[ind] = ii
    return map_inds, inv_map_inds, mask

# a function for solving the IB problem for a given beta, with multiple random initializations
def solve_IB(beta, p_XgY, p_Y, iterlimit=100000, init='random', N_inits=3, betastar=None, base_seed=209, ib_num=0):

    rng = np.random.default_rng(base_seed + ib_num)

    N_y = p_XgY.shape[1]
    N_x = p_XgY.shape[0]

    if p_Y is None:
        p_Y = np.array([1/N_y]*N_y)

    p_XY = p_XgY * p_Y
    p_X = np.sum(p_XY, axis=1, keepdims=True)
    p_YgX = p_XY / p_X

    IB_fxn_vals = []
    p_RgXs = []
    p_Rs = []
    p_YgRs = []
    for ii in range(N_inits):

        p_R = (np.ones(N_y) / N_y).reshape(1,-1)
        if init == 'random':
            p_RX = np.exp(rng.random((N_x,N_y)))
            p_RgX = p_RX / np.sum(p_RX,axis=1,keepdims=True)
        elif init == 'softmax':
            p_RgX = np.exp(betastar*p_YgX) / np.sum(np.exp(betastar*p_YgX),axis=1,keepdims=True)
        p_XgR = (p_RgX * p_X) / p_R
        p_YgR =  p_YgX.T @ p_XgR
        IB_fxn = compute_IB_fn(beta,p_RgX,p_XgY,p_Y)
        delta = 1
        iters = 0
        while (delta > 1e-15) or (iters < 100):
            # store last copy
            IB_fxn_last = IB_fxn

            # update p(r|x)
            p_RX = p_R * np.exp(beta*(p_YgX @ np.log(p_YgR))) # we can simplify the original update rule (below) to this
            # p_RX = p_R * np.exp(-beta*np.sum(p_YgX[:,:,np.newaxis] * (np.log(p_YgX[:,:,np.newaxis]) - np.log(p_YgR[np.newaxis,:,:])),axis=1))
            p_RgX = p_RX / np.sum(p_RX,axis=1,keepdims=True)

            # update p(r)
            p_R = p_X.T @ p_RgX

            #update p(y|r)
            p_XgR = (p_RgX * p_X) / p_R
            p_YgR =  p_YgX.T @ p_XgR

            IB_fxn = compute_IB_fn(beta,p_RgX,p_XgY,p_Y)

            delta = np.abs(IB_fxn - IB_fxn_last)
            iters += 1
            if iters == iterlimit:
                print(f"Warning: IB algorithm did not converge after {iterlimit} iterations for beta={beta:.4f}")
                break
        IB_fxn_vals.append(IB_fxn.copy())
        p_RgXs.append(p_RgX.copy())
        p_Rs.append(p_R.copy())
        p_YgRs.append(p_YgR.copy())

    min_ind = np.argmin(np.array(IB_fxn_vals))
    p_RgX = p_RgXs[min_ind]
    p_R = p_Rs[min_ind]
    p_YgR = p_YgRs[min_ind]
    # p_RY = p_YgR * p_R

    H_R = -np.sum(p_R * np.log2(p_R))
    I_XR = compute_I_XR(p_RgX,p_XgY,p_Y)
    I_RY = compute_I_RY(p_RgX,p_XgY,p_Y)

    return [I_XR, I_RY, beta, H_R, p_RgX, p_YgR, p_R, betastar]

# a function for solving the IB problem for a given beta, with multiple random initializations
# this is function is intended to be used within a multiprocessing call, so it takes in all of the arguments as a single tuple
def solve_IB_mp(args):
    beta, p_XgY, p_Y, iterlimit, init, N_inits, betastar, base_seed, ib_num = args
    return solve_IB(beta, p_XgY, p_Y, iterlimit=iterlimit, init=init, N_inits=N_inits, betastar=betastar, base_seed=base_seed, ib_num=ib_num)

# beta, p_XgY, p_Y, iterlimit, init, N_inits, betastar, base_seed, ib_num = args
def get_IB_bound(p_XgY,p_Y,N_b=1000,max_b=50,iterlimit=100000,init='random', N_inits=3, betastar=None, N_threads=1, base_seed=209):
    beta_array = np.linspace(max_b/N_b,max_b,N_b)

    with mp.Pool(processes=N_threads) as pool:
        results = [pool.apply_async(solve_IB_mp, args=((beta_array[ib], p_XgY, p_Y, iterlimit, init, N_inits, betastar, base_seed, ib),)) for ib in range(N_b)]
        results = [res.get() for res in results]
    I_XR = [res[0] for res in results]
    I_RY = [res[1] for res in results]
    betas = [res[2] for res in results]
    H_Rs = [res[3] for res in results]

    # to ensure that the points on the IB curve are in order of increasing beta, we can sort the results by beta before returning
    sort_indices = np.argsort(betas)
    return np.array(I_XR)[sort_indices], np.array(I_RY)[sort_indices], np.array(betas)[sort_indices], np.array(H_Rs)[sort_indices]

# get the IB measures for the IB optimal choice policy (computed using the true p(X,Y)) for a particular trial set (empirical p(X,Y))
def get_IB_emp(beta,Xemp,Xset,Yemp,p_XgY_true,p_Y_true, iterlimit=100000, init='random', N_inits=3, betastar=None, base_seed=209, ib_num=0):

    N_emp = Xemp.shape[0]
    Yemp_oh = np.eye(N_Y)[Yemp.astype(int)]
    p_Y_emp = np.sum(Yemp_oh,axis=0,keepdims=True) / N_emp

    Xinds = np.zeros(Xemp.shape[0], dtype=int)
    for ii, x in enumerate(Xset):
        matches = np.all(Xemp == x, axis=1)
        Xinds[matches] = ii

    out = solve_IB(beta, p_XgY_true, p_Y_true, iterlimit=iterlimit, init=init, N_inits=N_inits, betastar=betastar, base_seed=base_seed, ib_num=ib_num)
    p_RgX = out[4]
    p_RgX_emp = p_RgX[Xinds,:]
    p_RX_emp = p_RgX_emp * (1/N_emp)
    p_R_emp = np.sum(p_RX_emp, axis=0, keepdims=True)
    p_XgR_emp = p_RX_emp / p_R_emp
    p_YgR_emp = Yemp_oh.T @ p_XgR_emp
    I_XR_emp = compute_I_XR_emp(p_RgX_emp)
    I_RY_emp = compute_I_RY_emp(p_RgX_emp, Yemp)
    H_R_emp = -np.sum(p_R_emp * np.log2(p_R_emp))
    return I_XR_emp, I_RY_emp, beta, H_R_emp, p_RgX_emp, p_YgR_emp, p_R_emp, betastar

# same as above, but for multiprocessing
def get_IB_emp_mp(args):
    return get_IB_emp(*args)

def get_IB_bound_emp(Xemp,Xset,Yemp,p_XgY,p_Y,N_b=1000,max_b=50,iterlimit=100000,init='random', N_inits=3, betastar=None, N_threads=1, base_seed=209):
    beta_array = np.linspace(max_b/N_b,max_b,N_b)

    with mp.Pool(processes=N_threads) as pool:
        results = [pool.apply_async(get_IB_emp_mp, args=((beta_array[ib], Xemp, Xset, Yemp, p_XgY, p_Y, iterlimit, init, N_inits, betastar, base_seed, ib),)) for ib in range(N_b)]
        results = [res.get() for res in results]
    I_XR_emp = [res[0] for res in results]
    I_RY_emp = [res[1] for res in results]
    betas = [res[2] for res in results]
    H_Rs = [res[3] for res in results]

    # to ensure that the points on the IB curve are in order of increasing beta, we can sort the results by beta before returning
    sort_indices = np.argsort(betas)
    return np.array(I_XR_emp)[sort_indices], np.array(I_RY_emp)[sort_indices], np.array(betas)[sort_indices], np.array(H_Rs)[sort_indices]

# get IB measures for the softmax policy on the true p(X,Y)
def get_softmax_IB(betastar,p_XgY,p_Y):
   
    p_XY = p_XgY * p_Y
    p_X = np.sum(p_XY, axis=1, keepdims=True)
    p_YgX = p_XY / p_X

    p_RgX = np.exp(betastar*p_YgX) / np.sum(np.exp(betastar*p_YgX),axis=1,keepdims=True)
    p_XR = p_RgX*p_X
    p_R = np.sum(p_XR, axis=0, keepdims=True)
    p_XgR = (p_RgX * p_X) / p_R
    p_YgR =  p_YgX.T @ p_XgR

    accuracy = np.trace(p_YgR * p_R)
    I_XR = compute_I_XR(p_RgX,p_XgY,p_Y)
    I_RY = compute_I_RY(p_RgX,p_XgY,p_Y)
    H_R = -np.sum(p_R * np.log2(p_R))
    return I_XR, I_RY, betastar, H_R, p_RgX, p_YgR, p_R

def get_softmax_IB_bound(p_XgY,p_Y,N_b=1000,max_b=50):
    betastar_array = np.linspace(max_b/N_b,max_b,N_b)
    I_XRs = [], I_RYs = [], H_Rs = [], accuracies = []
    for betastar in betastar_array:
        out = get_softmax_IB(betastar,p_XgY,p_Y)
        I_XRs.append(out[0])
        I_RYs.append(out[1])
        H_Rs.append(out[3])
        accuracy = np.trace(out[5] * out[6])
        accuracies.append(accuracy)
    accuracies = np.array(accuracies)
    alphastars = np.log(accuracies/((1-accuracies)/(N_y-1))) # alpha* and this expression itself is only valid when p(Y) is uniform
    betas = betastar_array / alphastars
    return np.array(I_XRs), np.array(I_RYs), betastar_array, np.array(H_Rs), accuracies, alphastars, betas

def get_softmax_IB_bound_emp()

# compute likelihood distributions for the horse prediction task
def llr2probs_4shapes(llr,p1):
    
    # INPUT: array of llr for each of the 4 shapes, plus a parameter p1
    # OUTPUT: pdist1 is P(shapes|state 1), pdist2 is P(shapes|state 2)
    
    # p1 is the likelihood of shape 1 given state 1 (blue bar on the far left)
    pdist1 = np.zeros((4))
    pdist2 = np.zeros((4))

    lr = 10**llr

    pdist1[0] = p1
    pdist2[0] = lr[3]*p1
    pdist1[3] = pdist2[0]
    pdist2[3] = p1

    pdist1[1] = (1-(lr[3]+1)*p1) / (lr[2]+1)
    pdist2[1] = lr[2]*pdist1[1]
    pdist1[2] = pdist2[1]
    pdist2[2] = pdist1[1]

    return pdist1, pdist2

# convert a an array of shape combination codes to array of 4 integers, where each integer is the count of a particular shape in the combination
def split_to_four_digits(arr):
    # order of input array is [shape4, shape3, shape2, shape1]
    # need to convert to [shape1, shape2, shape3, shape4] eventually
    arr = arr.astype(int)
    result = np.zeros((arr.shape[0], 4), dtype=int)
    for i, num in enumerate(arr):
        digits = list(str(num).zfill(4))
        result[i] = [int(d) for d in digits]
    result = result[:, ::-1]  # reverse the order to get [shape1, shape2, shape3, shape4]
    return result

# compute likelihood of shape combination given horse
def P_shapecomb_g_horse(X, base, multiple, p1):
    optimal_llr = np.array([-base*multiple, -base, base, base*multiple]) 
    pdist1, pdist2 = llr2probs_4shapes(optimal_llr, p1)
    unique_perms = factorial(5) / np.array([factorial(X[ii]).prod() for ii in range(X.shape[0])])
    return np.column_stack(((pdist1**X).prod(axis=1)*unique_perms,(pdist2**X).prod(axis=1)*unique_perms))

# compute likelihood of shape combination given horse, with equal weighting of shapes 1&2 and shapes 3&4
def P_shapecomb_g_horse_ew(X, base, multiple, p1):
    optimal_llr = np.array([-base*multiple, -base, base, base*multiple]) 
    pdist1, pdist2 = llr2probs_4shapes(optimal_llr, p1)
    pdist1_ew = np.array([pdist1[0]+pdist1[1], pdist1[0]+pdist1[1], pdist1[2]+pdist1[3],  pdist1[2]+pdist1[3]])
    pdist2_ew = np.array([pdist2[0]+pdist2[1], pdist2[0]+pdist2[1], pdist2[2]+pdist2[3],  pdist2[2]+pdist2[3]])
    unique_perms = factorial(5) / np.array([factorial(X[ii][0]+X[ii][1]) * factorial(X[ii][2]+X[ii][3]) for ii in range(X.shape[0])])
    return np.column_stack(((pdist1_ew**X).prod(axis=1)*unique_perms,(pdist2_ew**X).prod(axis=1)*unique_perms))

# compute likelihood of shape combination given horse, with ignoring the weak shapes (shapes 2 and 3) when strong shapes (shapes 1 and 4) are present
def P_shapecomb_g_horse_iw(X, base, multiple, p1):
    optimal_llr = np.array([-base*multiple, -base, base, base*multiple]) 
    pdist1, pdist2 = llr2probs_4shapes(optimal_llr, p1)
    Xstrong = X.copy()
    Xstrong[:,1] = 0
    Xstrong[:,2] = 0
    Xiw = np.array([X[ii] if (X[ii,0]==0 and X[ii,3]==0) else Xstrong[ii] for ii in range(X.shape[0])])
    unique_perms = np.array([factorial(Xiw[ii].sum()) for ii in range(Xiw.shape[0])]) / np.array([factorial(Xiw[ii]).prod() for ii in range(Xiw.shape[0])])
    return np.column_stack(((pdist1**Xiw).prod(axis=1)*unique_perms,(pdist2**Xiw).prod(axis=1)*unique_perms))

# compute posterior probability of horse given shape combination
def P_horse_g_shapecomb(X, base, multiple, p1, P_XgY=None, p_Y=None):
    if P_XgY is None:
        p_XgY = P_shapecomb_g_horse(X, base, multiple, p1)
    else:
        p_XgY = P_XgY(X, base, multiple, p1)
    N_y = p_XgY.shape[1]

    if p_Y is None:
        p_Y = np.array([1/N_y]*N_y)

    p_XY = p_XgY * p_Y
    p_X = np.sum(p_XY, axis=1, keepdims=True)
    return p_XY / p_X