#! /usr/bin/env python3

from datetime import date

from loan import Loan
from loan_repayment import *
from utilities import print_schedules

###################################################################################
#  Main
###################################################################################
if __name__ == '__main__':
    test = Loan(
            principal=4500.00,
            apr=0.040,
            min_monthly_payment=55.00,
            interest=5.00,
            deferment_monthly_payment=00.00,
            start_date=date(2018, 12, 16),
            deferment_exit_date=date(2019, 6, 16),
            term=110)
    test2 = Loan(
            principal=6000.00,
            apr=0.055,
            min_monthly_payment=70.00,
            deferment_monthly_payment=00.00,
            start_date=date(2018, 12, 16),
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

    ############# Print Results ###################
    sep = '-'*80
    print("{}\nHIGH INTEREST\n{}".format(sep, sep))
    print_schedules(dfs_hi_int)
    print("{}\nLOW BALANCE\n{}".format(sep, sep))
    print_schedules(dfs_low_bal)

    print("---TOTAL COSTS---")
    print("\tMonthly Payment      : ${:0,.2f}".format( max_payment ))
    print("\tDefault              : ${:0,.2f}".format( test.default_cost() + test2.default_cost() ))
    print("\tHighest Interest Sim : ${:0,.2f}".format( float(cost_high_int_sim) ))
    print("\tLowest Balance Sim   : ${:0,.2f}".format( float(cost_low_bal_sim)))

