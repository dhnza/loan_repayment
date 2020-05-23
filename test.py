#! /usr/bin/env python3

from datetime import date

from loan import Loan
from loan_repayment import *
from minimize_cost import *
from utilities import print_schedules

###################################################################################
#  Main
###################################################################################
if __name__ == '__main__':
    test = Loan(
            principal=4500.00,
            apr=0.040,
            min_monthly_payment=49.00,
            deferment_monthly_payment=00.00,
            start_date=date(2018, 12, 26),
            deferment_exit_date=date(2019, 6, 16),
            term=110)
    test2 = Loan(
            principal=6000.00,
            apr=0.055,
            min_monthly_payment=70.00,
            deferment_monthly_payment=00.00,
            start_date=date(2018, 12, 26),
            deferment_exit_date=date(2019, 6, 16),
            term=110)

    loans = [test, test2]
    max_payment=150

    # Simulate high interest schedule
    lrp_hi_int = LoanRepayment(loans, max_payment)
    dfs_hi_int = lrp_hi_int.schedule_max_interest()
    cost_high_int_sim = sum( df['Payment'].sum() for df in dfs_hi_int )

    # Simulate low balance schedule
    lrp_low_bal = LoanRepayment(loans, max_payment)
    dfs_low_bal = lrp_low_bal.schedule_lowest_balance()
    cost_low_bal_sim = sum( df['Payment'].sum() for df in dfs_low_bal )

    # Default repayment schedule
    default_cost = sum(loan.default_cost() for loan in loans)

    # Optimal schedule
    # dfs_min = minimize_cost(loans, max_payment)
    # cost_min = sum( df['Payment'].sum() for df in dfs_min )

    ############# Print Results ###################
    sep = '-'*80
    print("{}\nHIGH INTEREST\n{}".format(sep, sep))
    print_schedules(dfs_hi_int)
    print("{}\nLOW BALANCE\n{}".format(sep, sep))
    print_schedules(dfs_low_bal)
    # print("{}\nOPTIMAL\n{}".format(sep, sep))
    # print_schedules(dfs_min)

    print("---TOTAL COSTS---")
    print("\tMonthly Payment      : ${:0,.2f}".format(max_payment))
    print("\tDefault              : ${:0,.2f}".format(default_cost))
    print("\tHighest Interest Sim : ${:0,.2f}".format(cost_high_int_sim))
    print("\tLowest Balance Sim   : ${:0,.2f}".format(cost_low_bal_sim))
    # print("\tOptimized Sim        : ${:0,.2f}".format(cost_min))

    # dfs_min = minimize_cost(loans, max_payment)
    dfs_min = minimize_cost2(loans, max_payment)
    # dfs_min = minimize_cost3(loans, max_payment)
    # dfs_min = minimize_cost4(loans, max_payment)
    # dfs_min = minimize_cost_global(loans, max_payment)

    cost_min = sum( df['Payment'].sum() for df in dfs_min )
    print("{}\nOPTIMAL\n{}".format(sep, sep))
    print_schedules(dfs_min)
    print("\n\tOptimized Sim   : ${:0,.2f}".format(cost_min))
