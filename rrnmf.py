import numpy as np

def RRNMF(M, rank=25):
    """
    Factorize M = W * H
    Assume every column of W is exactly the same as a column in M
    Without loss of generality assume every column of M sums to 1.
    rank: int
    """

    # Normalize columns of M
    for i in xrange(M.shape[1]):
        M[:,i] = M[:,i] / np.linalg.norm(M[:,i])
    
    R, J = M, set()
    while len(J) < rank:
        col_norms = np.linalg.norm(R, axis=0)
        j = np.argmax(col_norms)
        J.add(j)
        u_j = R[:,j].reshape((R.shape[0],1))
        R = np.dot(np.identity(R.shape[0])-np.dot(u_j,np.transpose(u_j))/np.linalg.norm(u_j),R)

    J = sorted(list(J))
    W = M[:,J]
    H = np.zeros((rank,M.shape[1]))
    j = 0
    for i in xrange(M.shape[1]):
        if i in J:
            #The i-th column of M is exactly the j-th column of W:
            H[j,i] = 1
            j += 1
        else:
            # We have: M[:,i] = W * H[:,i]
            H[:,i], residuals, _, _ = np.linalg.lstsq(W,M[:,i])
        #print np.linalg.norm(M[:,i]-np.dot(W,H[:,i]))


    return J, W, H



if __name__ == "__main__":
    dim1, dim2, rank = 200, 100, 25
    W = np.random.sample((dim1,rank))
    #for i in xrange(rank):
    #    W[:,i] = W[:,i] / np.sum(W[:,i])

    H = np.concatenate((np.identity(rank),np.random.sample((rank,dim2-rank))),axis=1)
    #for i in xrange(rank,dim2):
    #    H[:,i] = H[:,i] / np.sum(H[:,i])

    M = np.dot(W,H)

    permutation = np.random.permutation(dim2)
    ground_truth = permutation[:25]
    indexes = np.argsort(permutation)
    M = M[:,indexes]

    result, W, H = RRNMF(M,rank)

    print sorted(ground_truth)
    print sorted(result)
    
    for i in xrange(M.shape[1]):
        M[:,i] = M[:,i] / np.linalg.norm(M[:,i])
    print("L2_norm(M-WH)/L2_norm(M): {0}".format(np.linalg.norm(M - np.dot(W,H))/np.linalg.norm(M)))
