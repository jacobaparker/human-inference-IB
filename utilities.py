import numpy as np
import ndd
from math import factorial

### Functions for performing IB analyses on empirical data ###

# compute mutual information using NSB method
def mutual_inf_nsb(x,y,ks):
    """
    Calculate mutual information using NSB method
    """
    ar = np.column_stack((x,y))
    mi = ndd.mutual_information(ar,ks)
    return np.log2(np.e)*mi #ndd returns nats - multiply by log2(e) to convert to bits

### Functions for computing IB bounds/curves ###

def compute_I_XR(p_RgX,p_X,p_R):
    return np.sum(p_RgX*p_X*np.log2(p_RgX/p_R))

def compute_I_RY(p_YgR,p_R,p_Y):
    return np.sum(p_YgR*p_R*np.log2(p_YgR/p_Y.reshape(-1,1)))

def compute_I_XR_emp(p_RgX_emp,p_R_emp):
    return np.sum(p_RgX_emp * (1/p_RgX_emp.shape[0])*np.log2(p_RgX_emp/p_R_emp))

def compute_I_RY_emp(p_YgR_emp,p_R_emp,p_Y_emp):
    return np.sum(p_YgR_emp * p_R_emp * np.log2(p_YgR_emp/p_Y_emp.reshape(-1,1)))

def compute_IB_fn(beta,p_RgX,p_X,p_R,p_YgR,p_Y):
        Ixr = compute_I_XR(p_RgX,p_X,p_R)
        Iry = compute_I_RY(p_YgR,p_R,p_Y)
        return Ixr - beta*Iry

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
def solve_IB(beta, p_XgY, p_Y, iterlimit=100000, init='random', N_inits=3, betastar=None):
    rng = np.random.default_rng()

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
        IB_fxn = compute_IB_fn(beta,p_RgX,p_X,p_R,p_YgR,p_Y)
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

            IB_fxn = compute_IB_fn(beta,p_RgX,p_X,p_R,p_YgR,p_Y)

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
    p_RY = p_YgR * p_R

    H_R = -np.sum(p_R * np.log2(p_R))
    I_XR = compute_I_XR(p_RgX,p_X,p_R)
    I_RY = compute_I_RY(p_YgR,p_R,p_Y)

    return [I_XR, I_RY, beta, H_R, p_RgX, p_YgR, betastar]

# a function for solving the IB problem for a given beta, with multiple random initializations
# this is function is intended to be used within a multiprocessing call, so it takes in all of the arguments as a single tuple
def solve_IB_mp(args):
    beta, p_XgY, p_Y, iterlimit, init, N_inits, betastar, base_seed, ib_num = args

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
        IB_fxn = compute_IB_fn(beta,p_RgX,p_X,p_R,p_YgR,p_Y)
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

            IB_fxn = compute_IB_fn(beta,p_RgX,p_X,p_R,p_YgR,p_Y)

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
    p_RY = p_YgR * p_R

    H_R = -np.sum(p_R * np.log2(p_R))
    I_XR = compute_I_XR(p_RgX,p_X,p_R)
    I_RY = compute_I_RY(p_YgR,p_R,p_Y)

    return [I_XR, I_RY, beta, H_R, p_RgX, p_YgR, betastar]

# get the IB measures for the IB optimal choice policy for a particular trial set ("empirical data")
def get_IB_emp(beta,Xemp,X,Yemp,P_XgY,p_Y, iterlimit=100000, init='random', N_inits=3, betastar=None):
    out = solve_IB(beta, p_XgY, p_Y, iterlimit=100000, init='random', N_inits=3, betastar=None)

# compute the 
# def compute_IB_bound(p_XgY, p_Y, betas, iterlimit=100000, init='random', N_inits=3, betastar=None):

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