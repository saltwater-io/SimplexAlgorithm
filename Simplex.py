# Simplex Algorithm adapted from tutorial here:
#  https://medium.com/@jacob.d.moore1/coding-the-simplex-algorithm-from-scratch-using-python-and-numpy-93e3813e6e70
#  Uses the big M method
#
# CSC 320: Linear Programming - Mrs. Elliot
#               Written by: Dakota McGuire

import numpy as np


# Generates a matrix that fits all needs
# Including slack variables w/ 0 fillers
def gen_matrix(var, cons):
    mat = np.zeros((cons + 1, var + cons + 2))
    return mat


# Checks to see if 1+ pivots are needed due
#  to negative element(s) in furthest right column
def next_pivot_right(table):
    m = min(table[:-1, -1])
    if m >= 0:
        return False
    else:
        return True


# Checks to see if 1+ pivots are needed due
#  to negative element(s) in bottom row
def next_pivot(table):
    lr = len(table[:, 0])
    m = min(table[lr - 1, :-1])
    if m >= 0:
        return False
    else:
        return True


# Returns index of pivot on right column
def find_pivot_right(table):
    lc = len(table[0, :])
    m = min(table[:-1, lc - 1])
    if m <= 0:
        n = np.where(table[:-1, lc - 1] == m)[0][0]
    else:
        n = None
    return n


# Returns index of pivot on bottom row
def find_neg(table):
    lr = len(table[:, 0])
    m = min(table[lr - 1, :-1])
    if m <= 0:
        n = np.where(table[lr - 1, :-1] == m)[0][0]
    else:
        n = None
    return n


# Find pivot value in the right columns row
# Most negative value in row
def loc_piv_right(table):
    total = []
    r = find_pivot_right(table)
    row = table[r, :-1]
    m = min(row)
    c = np.where(row == m)[0][0]
    col = table[:-1, c]
    for i, b in zip(col, table[:-1, -1]):
        if i ** 2 > 0 and b / i > 0:
            total.append(b / i)
        else:
            total.append(10000)
    index = total.index(min(total))
    return [index, c]


# Find pivot value in the bottom row's column
# Most negative value in column
def loc_piv(table):
    if next_pivot(table):
        total = []
        n = find_neg(table)
        for i, b in zip(table[:-1, n], table[:-1, -1]):
            if b / i > 0 and i ** 2 > 0:
                total.append(b / i)
            else:
                total.append(10000)
        index = total.index(min(total))
        return [index, n]


# Converts tables to support minimization problems
# This is using big M method, so objective function is left "as is"
def convert_min(table):
    table[-1, :-2] = [-1 * i for i in table[-1, :-2]]
    table[-1, -1] = -1 * table[-1, -1]
    return table


# Pivot function to remove negative entry in the final column or row
# then updates table
def pivot(row, col, table):
    lr = len(table[:, 0])
    lc = len(table[0, :])
    t = np.zeros((lr, lc))
    pr = table[row, :]
    if table[row, col] ** 2 > 0:
        e = 1 / table[row, col]
        r = pr * e
        for i in range(len(table[:, col])):
            k = table[i, :]
            c = table[i, col]
            if list(k) == list(pr):
                continue
            else:
                t[i, :] = list(k - r * c)
        t[row, :] = list(r)
        return t
    else:
        print('Cannot pivot on this element.')


# Generates a table w/ x variables
def gen_var(table):
    lc = len(table[0, :])
    lr = len(table[:, 0])
    var = lc - lr - 1
    v = []
    for i in range(var):
        v.append('x' + str(i + 1))
    return v


# Determines if constraints need to be added
def add_cons(table):
    lr = len(table[:, 0])
    empty = []
    for i in range(lr):
        total = 0
        for j in table[i, :]:
            total += j ** 2
        if total == 0:
            empty.append(total)
    if len(empty) > 1:
        return True
    else:
        return False


# determines if objective function can be added
def add_obj(table):
    lr = len(table[:, 0])
    empty = []
    for i in range(lr):
        total = 0
        for j in table[i, :]:
            total += j ** 2
        if total == 0:
            empty.append(total)
    if len(empty) == 1:
        return True
    else:
        return False


# Adds constraints from equation into tableau
def constrain(table, eq):
    if add_cons(table):
        lc = len(table[0, :])
        lr = len(table[:, 0])
        var = lc - lr - 1
        j = 0
        while j < lr:
            row_check = table[j, :]
            total = 0
            for i in row_check:
                total += float(i ** 2)
            if total == 0:
                row = row_check
                break
            j += 1
        eq = convert(eq)
        i = 0
        while i < len(eq) - 1:
            row[i] = eq[i]
            i += 1
        row[-1] = eq[-1]
        row[var + j] = 1
    else:
        print('Cannot add another constraint.')


# Adds objective function to last row of tableau
def obj(table, eq):
    if add_obj(table):
        eq = [float(i) for i in eq.split(',')]
        lr = len(table[:, 0])
        row = table[lr - 1, :]
        i = 0
        while i < len(eq) - 1:
            row[i] = eq[i] * -1
            i += 1
        row[-2] = 1
        row[-1] = eq[-1]
    else:
        print('You must finish adding constraints before the objective function can be added.')


# Runs maximization function
def max_z(table):
    while next_pivot_right(table):
        table = pivot(loc_piv_right(table)[0], loc_piv_right(table)[1], table)
    while next_pivot(table):
        table = pivot(loc_piv(table)[0], loc_piv(table)[1], table)
    lc = len(table[0, :])
    lr = len(table[:, 0])
    var = lc - lr - 1
    val = {}
    for i in range(var):
        col = table[:, i]
        s = sum(col)
        m = max(col)
        if float(s) == float(m):
            loc = np.where(col == m)[0][0]
            val[gen_var(table)[i]] = table[loc, -1]
        else:
            val[gen_var(table)[i]] = 0
    val['max'] = table[-1, -1]
    return val


# Runs minimization function
def min_z(table):
    table = convert_min(table)
    while next_pivot_right(table):
        table = pivot(loc_piv_right(table)[0], loc_piv_right(table)[1], table)
    while next_pivot(table):
        table = pivot(loc_piv(table)[0], loc_piv(table)[1], table)
    lc = len(table[0, :])
    lr = len(table[:, 0])
    var = lc - lr - 1
    val = {}
    for i in range(var):
        col = table[:, i]
        s = sum(col)
        m = max(col)
        if float(s) == float(m):
            loc = np.where(col == m)[0][0]
            val[gen_var(table)[i]] = table[loc, -1]
        else:
            val[gen_var(table)[i]] = 0
            val['min'] = table[-1, -1] * -1
    return val


# Converts a variable 'G' or 'L"
# for greater than/less than
# Respectively
def convert(eq):
    eq = eq.split(',')
    if 'G' in eq:
        g = eq.index('G')
        del eq[g]
        eq = [float(i) * -1 for i in eq]
        return eq
    if 'L' in eq:
        l = eq.index('L')
        del eq[l]
        eq = [float(i) for i in eq]
        return eq


# Runs program
def main():
    # Currently set to test #3 from Ch4 HW pt. 2
    # generates matrix
    m = gen_matrix(3, 3)
    # Adds constraints
    constrain(m, '3,1,1,L,60')
    constrain(m, '2,1,2,L,20')
    constrain(m, '2,2,1,L,20')
    # Adds objective function
    obj(m, '2,-1,1,0')
    print(max_z(m))

    # m = gen_matrix(2, 4)
    # constrain(m, '2,5,G,30')
    # constrain(m, '-3,5,G,5')
    # constrain(m, '8,3,L,85')
    # constrain(m, '-9,7,L,42')
    # obj(m, '2,7,0')
    # print(min_z(m))


if __name__ == "__main__":
    main()
