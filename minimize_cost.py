import math
import numpy as np
from datetime import date, timedelta
from scipy.optimize import linprog, shgo, differential_evolution, LinearConstraint, NonlinearConstraint

from utilities import isclose

# Objective function
def obj_fun(x):
    return np.sum(x)

def status(xk, convergence):
    print(f'CONVERGENCE: {convergence}')
    print(f'f: {obj_fun(xk)}')
    # print(f'X_k: {xk}')
    return convergence > 0.95

def minimize_cost_global(loans, max_payment):

    t_max = max(( loan.term for loan in loans )) # maximum terms
    n_l   = len(loans)                           # number of loans
    N     = t_max*n_l                            # dimension of x
    cons = []

    # TODO: this is a field of LoanRepayment
    min_payment = sum([loan.min_mp for loan in loans])

    ###### Construct linear constraints ######
    n_cons = n_l + t_max  # number of constraints
    A = np.zeros((n_cons, N))
    b_lb = np.zeros(n_cons)
    b_ub = np.zeros(n_cons)

    # Repayment condition
    for l, loan in enumerate(loans):
        it = 1.0  # interest
        for t in range(t_max-1, -1, -1):
            it *= (1 + loan.mpr)
            A[l, t*n_l + l] = it

        b_lb[l] = it * loan.principal
        # b_ub[l] = it * loan.principal
        b_ub[l] = it * (loan.principal * (1 + loan.mpr))

    # Maximum payment condition
    for t in range(t_max):
        for l in range(n_l):
            A[n_l + t, t*n_l + l] = 1.0

        b_lb[n_l + t] = 0
        b_ub[n_l + t] = max_payment

    cons.append(LinearConstraint(A, b_lb, b_ub))
    print(f'A: {A}')
    print(f'b_lb: {b_lb}')
    print(f'b_ub: {b_ub}')

    ###### Construct non-linear constraints ######
    b_lb_nlc = np.zeros(n_l)
    b_ub_nlc = np.zeros(n_l)
    # Minimum payment condition
    def nlc(x):
        ret = np.zeros(n_l)

        # for l, loan in enumerate(loans):
            # for t in range(0, t_max):
                # ret[t*n_l+l] = (x[t*n_l+l] - loan.min_mp) if ((np.dot(A[l,0:t+1], x[0:t+1]) - b_lb[l]) < 0) and (x[t*n_l+l] < loan.min_mp) else 0

        for l, loan in enumerate(loans):
            payments = x[l:N:n_l]
            mask = payments < loan.min_mp
            mask &= abs(payments-loan.min_mp) > 1
            idx = np.where(mask)[0]
            if len(idx) == 0:
                ret[l] = 0
                continue

            first_zero = n_l*idx[0] + l
            test_x = np.zeros(N)
            test_x[0:first_zero+1] = x[0:first_zero+1]
            left = b_lb[l] - np.dot(A[l], test_x)
            if (left > 0):
                ret[l] = loan.min_mp - x[first_zero+1]
            else:
                ret[l] = 0

        return ret

    cons.append(NonlinearConstraint(nlc, b_lb_nlc, b_ub_nlc))

    ###### Contruct bounds on variables ######
    bounds = [(0, max_payment)]*N

    # for l, loan in enumerate(loans):
    #     mp = loan.mp
    #     loan.mp = max_payment - min_payment + loan.min_mp
    #     min_term = math.floor(loan.months_to_repayment())
    #     loan.mp = mp
    #
    #     for t in range(min_term):
    #         bounds[t*n_l + l] = (loan.min_mp, max_payment)

    ###### Run global minimization problem ######
    res = differential_evolution(obj_fun, bounds, constraints=cons, workers=-1, callback=status, disp=True, maxiter=10)
    # res = differential_evolution(obj_fun, bounds, workers=-1, callback=status, disp=True)
    print(res)

    Ax = A.dot(res.x)
    print(f'A.x: {Ax}')
    print(f'A.x <= b_ub: {np.less_equal(Ax, b_ub)}')
    print(f'A.x >= b_lb: {np.greater_equal(Ax, b_lb)}')
    print(f'nlc(x): {nlc(res.x)}')

    ###### Run repayment simulation with optimal schedule ######
    for l, loan in enumerate(loans):
        payments = [ res.x[t*n_l + l] for t in range(t_max)]
        loan.simulate_payments(payments)
        loan.mp = loan.min_mp
        loan.simulate_full_repayment()

    return [ loan.construct_df() for loan in loans ]


########################################## START MINIMIZE COST 2 #########################################
# Change lower bounds if non-zero before repayment

def minimize_cost2(loans, max_payment):

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

    # TODO: daily interest
    # for l, loan in enumerate(loans):
    #     prev = cur = date.today()
    #     it = 1.0  # interest
    #     for t in range(t_max-1, -1, -1):
    #         cur += loan.next_payment
    #         days = (cur - prev).days
    #         prev = cur
    #         it *= (1 + loan.dpr)**days
    #         A_eq[l, t*n_l + l] = it
    #
    #     b_eq[l] = it * loan.principal

    # TODO: monthly interest
    for l, loan in enumerate(loans):
        it = 1.0  # interest
        for t in range(t_max-1, -1, -1):
            it *= (1 + loan.mpr)
            A_eq[l, t*n_l + l] = it

        b_eq[l] = it * (loan.principal * (1 + loan.mpr))

    ###### Construct inequality constraints ######
    A_ub = np.zeros((t_max, N))
    b_ub = np.zeros(t_max)

    for t in range(t_max):
        for l in range(n_l):
            A_ub[t, t*n_l + l] = 1.0

        b_ub[t] = max_payment

    ###### Contruct bounds on variables ######
    # bounds = (0, max_payment)
    bounds = [(0, max_payment)]*N

    for l, loan in enumerate(loans):
        mp = loan.mp
        loan.mp = max_payment - min_payment + loan.min_mp
        min_term = math.floor(loan.months_to_repayment())
        loan.mp = mp

        for t in range(min_term):
            bounds[t*n_l + l] = (loan.min_mp, max_payment)

    ###### Continue until minimum payment constraint is satisfied ######
    done = False
    while not done:

        ###### Run linear optimization problem ######
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='revised simplex')

        done = True
        for l, loan in enumerate(loans):
            payments = res.x[l:N:n_l]
            mask = payments < loan.min_mp
            mask &= abs(payments-loan.min_mp) > 1
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue

            first_zero = n_l*idx[0] + l
            test_x = np.zeros(N)
            test_x[0:first_zero] = res.x[0:first_zero]
            left = b_eq[l] - np.dot(A_eq[l], test_x)

            if not isclose(left, 0, 1):
                bounds[first_zero] = (min(loan.min_mp, left), max_payment)
                done = False
                break

    ###### Run repayment simulation with optimal schedule ######
    for l, loan in enumerate(loans):
        payments = [ res.x[t*n_l + l] for t in range(t_max)]
        loan.simulate_payments(payments)
        loan.mp = loan.min_mp
        loan.simulate_full_repayment()

    return [ loan.construct_df() for loan in loans ]

########################################## END MINIMIZE COST 2 #########################################



########################################## START MINIMIZE COST 3 #########################################
# Attempt to add linear constraint for non-zeros

def minimize_cost3(loans, max_payment):

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

    # TODO: daily interest
    # for l, loan in enumerate(loans):
    #     prev = cur = date.today()
    #     it = 1.0  # interest
    #     for t in range(t_max-1, -1, -1):
    #         cur += loan.next_payment
    #         days = (cur - prev).days
    #         prev = cur
    #         it *= (1 + loan.dpr)**days
    #         A_eq[l, t*n_l + l] = it
    #
    #     b_eq[l] = it * loan.principal

    # TODO: monthly interest
    for l, loan in enumerate(loans):
        it = 1.0  # interest
        for t in range(t_max-1, -1, -1):
            it *= (1 + loan.mpr)
            A_eq[l, t*n_l + l] = it

        b_eq[l] = it * (loan.principal * (1 + loan.mpr))

    ###### Construct inequality constraints ######
    A_ub = np.zeros((t_max*(n_l+1), N))
    b_ub = np.zeros(t_max*(n_l+1))

    for t in range(t_max):
        for l in range(n_l):
            A_ub[t, t*n_l + l] = 1.0

        b_ub[t] = max_payment

    for l, loan in enumerate(loans):
        for t in range(1,t_max):
            for k in range(t):
                A_ub[(l+1)*t_max+t, k*n_l + l] = 1.0

            A_ub[(l+1)*t_max+t, t*n_l+l] = -t
            # print(f"A_ub[{(l+1)*t_max+t}] = {A_ub[(l+1)*t_max+t]}")

        b_ub[t_max+t] = 0

    print(A_ub)

    ###### Contruct bounds on variables ######
    # bounds = (0, max_payment)
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
    # print(res)

    ###### Run repayment simulation with optimal schedule ######
    for l, loan in enumerate(loans):
        payments = [ res.x[t*n_l + l] for t in range(t_max)]
        loan.simulate_payments(payments)
        loan.mp = loan.min_mp
        loan.simulate_full_repayment()

    return [ loan.construct_df() for loan in loans ]

########################################## END MINIMIZE COST 3 #########################################



########################################## START MINIMIZE COST 4 #########################################
# Change weights in objective function


def minimize_cost4(loans, max_payment):

    t_max = max(( loan.term for loan in loans )) # maximum terms
    n_l   = len(loans)                           # number of loans
    N     = t_max*n_l                            # dimension of x

    # TODO: this is a field of LoanRepayment
    min_payment = sum([loan.min_mp for loan in loans])

    # Coefficients of objective function
    c = np.ones(N)

    val = 1.0
    for t in range(t_max):
        c[n_l*t:n_l*(t+1)] = val
        val *= 1.5

    print(f'C: {c}')

    ###### Construct equality constraints ######
    A_eq = np.zeros((n_l, N))
    b_eq = np.zeros(n_l)

    for l, loan in enumerate(loans):
        prev = cur = date.today()
        it = 1.0  # interest
        for t in range(t_max-1, -1, -1):
            cur += loan.next_payment
            days = (cur - prev).days
            prev = cur
            it *= (1 + loan.dpr)**days
            A_eq[l, t*n_l + l] = it

        b_eq[l] = it * (loan.principal * (1 + loan.mpr))

    ###### Construct inequality constraints ######
    A_ub = np.zeros((t_max, N))
    b_ub = np.zeros(t_max)

    for t in range(t_max):
        for l in range(n_l):
            A_ub[t, t*n_l + l] = 1.0

        b_ub[t] = max_payment

    ###### Contruct bounds on variables ######
    # bounds = (0, max_payment)
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
    # print(res)

    ###### Run repayment simulation with optimal schedule ######
    for l, loan in enumerate(loans):
        payments = [ res.x[t*n_l + l] for t in range(t_max)]
        loan.simulate_payments(payments)
        loan.mp = loan.min_mp
        loan.simulate_full_repayment()

    return [ loan.construct_df() for loan in loans ]

########################################## END MINIMIZE COST 4 #########################################


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
