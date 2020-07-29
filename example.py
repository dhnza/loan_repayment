#! /usr/bin/env python3

from datetime import date
from dateutil.relativedelta import relativedelta

from loan import Loan
from loan_repayment import *
from minimize_cost import *
from utilities import print_schedules

TODAY = date.today()
# Deferment ends 6 months in the future
DEFERMENT_END = TODAY + relativedelta(months=+6)

LOANS = [
        Loan(
            principal=4500.00,
            apr=0.040,
            min_monthly_payment=55.00,
            deferment_monthly_payment=00.00,
            start_date=TODAY,
            deferment_exit_date=DEFERMENT_END,
            term=110),
        Loan(
            principal=6000.00,
            apr=0.055,
            min_monthly_payment=70.00,
            deferment_monthly_payment=00.00,
            start_date=TODAY,
            deferment_exit_date=DEFERMENT_END,
            term=110)
        ]

###################################################################################
#  Main
###################################################################################
if __name__ == '__main__':
    MAX_PAYMENT=200

    # Default repayment schedule
    default_cost = sum(loan.default_cost() for loan in LOANS)

    # Simulate high interest schedule
    lrp_hi_int = LoanRepayment(LOANS, MAX_PAYMENT)
    dfs_hi_int = lrp_hi_int.schedule_max_interest()
    cost_high_int_sim = sum( df['Payment'].sum() for df in dfs_hi_int )

    # Simulate low balance schedule
    lrp_low_bal = LoanRepayment(LOANS, MAX_PAYMENT)
    dfs_low_bal = lrp_low_bal.schedule_lowest_balance()
    cost_low_bal_sim = sum( df['Payment'].sum() for df in dfs_low_bal )

    # Optimal schedule
    dfs_min = minimize_cost(LOANS, MAX_PAYMENT)
    cost_min = sum( df['Payment'].sum() for df in dfs_min )

    ############# Print Results ###################
    header = "{0}\n{{}}\n{0}".format('-'*80)
    print(header.format("HIGH INTEREST"))
    print_schedules(dfs_hi_int)
    print(header.format("LOW BALANCE"))
    print_schedules(dfs_low_bal)
    print(header.format("OPTIMAL"))
    print_schedules(dfs_min)

    print("---TOTAL COSTS---")
    print("\tMonthly Payment      : ${:0,.2f}".format(MAX_PAYMENT))
    print("\tDefault              : ${:0,.2f}".format(default_cost))
    print("\tHighest Interest Sim : ${:0,.2f}".format(cost_high_int_sim))
    print("\tLowest Balance Sim   : ${:0,.2f}".format(cost_low_bal_sim))
    print("\tOptimized Sim        : ${:0,.2f}".format(cost_min))
