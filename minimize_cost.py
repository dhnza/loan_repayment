import math
import numpy as np
from scipy.optimize import linprog

def minimize_cost(loans, max_payment):

    t_max = max(( loan.term for loan in loans )) # maximum terms
    n_l   = len(loans)                           # number of loans
    N     = t_max*n_l                            # dimension of x

    # TODO: this is a field of LoanRepayment
    min_payment = sum([loan.min_mp for loan in loans])

    # Coefficients of objective function
    c = np.ones(N)

    ###### Construct equality constraints ######
    A_eq = np.zeros((n_l, N))
    b_eq = np.zeros(n_l)

    for l, loan in enumerate(loans):
        # TODO: make interest vary by month?
        it = 1.0  # interest
        for t in range(t_max-1, -1, -1):
            it *= (1 + loan.mpr)
            A_eq[l, t*n_l + l] = it

        b_eq[l] = it * loan.principal

    ###### Construct inequality constraints ######
    A_ub = np.zeros((t_max, N))
    b_ub = np.zeros(t_max)

    for t in range(t_max):
        for l in range(n_l):
            A_ub[t, t*n_l + l] = 1.0

        b_ub[t] = max_payment

    ###### Contruct bounds on variables ######
    bounds = [(0, max_payment)]*N

    for l, loan in enumerate(loans):
        mp = loan.mp
        loan.mp = max_payment - min_payment + loan.min_mp
        min_term = math.floor(loan.months_to_repayment())
        loan.mp = mp

        for t in range(min_term):
            bounds[t*n_l + l] = (loan.min_mp, max_payment)

    ###### Run linear optimization problem ######
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='revised simplex')

    ###### Run repayment simulation with optimal schedule ######
    for l, loan in enumerate(loans):
        payments = [ res.x[t*n_l + l] for t in range(t_max)]
        loan.simulate_payments(payments)

    return [ loan.construct_df() for loan in loans ]
