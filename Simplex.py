import numpy as np


# Generates a matrix that fits all needs
# Including slack variables w/ 0 fillers
def gen_matrix(var,cons):
    mat = np.zeros((cons+1, var+cons+2))
    return mat

def next_round_r(table):
    m = min(table[:-1,-1])
    if m>= 0:
        return False
    else:
        return True

def next_round(table):
    lr = len(table[:,0])
    m = min(table[lr-1,:-1])
    if m>=0:
        return False
    else:
        return True


def find_neg_r(table):
    lc = len(table[0,:])
    m = min(table[:-1,lc-1])
    if m<=0:
        n = np.where(table[:-1,lc-1] == m)[0][0]
    else:
        n = None
    return n

def find_neg(table):
    lr = len(table[:,0])
    m = min(table[lr-1,:-1])
    if m<=0:
        n = np.where(table[lr-1,:-1] == m)[0][0]
    else:
        n = None
    return n

def main():
    pass

if __name__=="__main__":
    main()
