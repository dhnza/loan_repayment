import math
import numpy as np
from copy import deepcopy
from scipy.optimize import linprog

#TODO: remove
import pandas as pd

def minimize_cost(loans, max_payment):

    t_max = max((loan.term +
                 loan.months_to_deferment_exit()
                 for loan in loans ))  # maximum terms
    n_l   = len(loans)                 # number of loans
    N     = t_max*n_l                  # dimension of x

    # TODO: this is a field of LoanRepayment
    min_payment = sum([loan.min_mp for loan in loans])

    ###### Minimum payment simulation ######
    loans_sim = deepcopy(loans)
    cum_int    = np.zeros(t_max)
    principals = np.zeros((n_l, t_max))
    interests  = np.zeros((n_l, t_max))

    for l, loan in enumerate(loans_sim):
        loan.mp = loan.min_mp
        loan.dmp = 0
        df, total_cost = loan.simulate_full_repayment()

        principals[l,:] = loan.init_principal
        principals[l,:df.shape[0]-1] = df.diff()['Principal'].cumsum()[1:].abs()

        cum_int[:] = loan.mpr + 1
        cum_int[:df.shape[0]-1] = df['Interest'][1:] + 1
        interests[l,:] = cum_int[::-1].cumprod()[::-1]

    # Coefficients of objective function
    c = np.ones(N)

    ###### Construct inequality constraints ######
    A_ub = np.zeros((N+t_max, N))
    b_ub = np.zeros(N+t_max)

    # Pay at least as much principal as minimum schedule
    for l, loan in enumerate(loans):
        for t in range(t_max):
            A_ub[t_max*l + t, t_max*l:t_max*l + t+1] = -interests[l,:t+1]

        b_ub[t_max*l:t_max*(l+1)] = -interests[l,0] * principals[l,:]

    # Pay at most maximum monthly payment
    for t in range(t_max):
        for l in range(n_l):
            A_ub[N+t, t_max*l + t] = 1.0

    b_ub[N:] = max_payment

    ###### Contruct bounds on variables ######
    bounds = (0, max_payment)

    ###### Run linear optimization problem ######
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    ###### Run repayment simulation with optimal schedule ######
    payments = res.x.reshape(n_l, t_max)
    for l, loan in enumerate(loans):
        loan.simulate_payments(payments[l,:])

    return [ loan.construct_df() for loan in loans ]
