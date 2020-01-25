from copy import deepcopy

from loan import Loan
from utilities import MinHeap


class LoanRepayment:


    def __init__(self, loans, max_payment):
        self.loans = deepcopy(loans)
        self.max_payment = max_payment

        self.min_payment = sum([loan.min_mp for loan in self.loans])
        if self.min_payment > self.max_payment:
            raise ValueError("Max payment of ${:0,.2f} is less than minimum loan payments of ${:0,.2f}".format(self.max_payment, self.min_payment))


    def balance_remains():
        """
        Check whether balance remains on any loan.
        """
        for loan in self.loans:
            if loan.balance > 0:
                return true
        return false


    def simulate_payment(self, n=1):
        """
        Simulates n payments on all loans.

        :param int n: The number of payments to simulate
        :return: Nothing, called for side-effect.
        """
        for loan in self.loans:
            loan.simulate_payment(n)


    # TODO: Assumes all loans are in deferment/repayment at the same time
    def schedule_prioritize_one(self, key):
        """
        Simulates full repayment on all loans. After allocating minimum monthly payments
        to each loan, any surplus is given to the loan with the minimal key.

        :param func key: determines the key by which to prioritize loans
        :return: dfs, a list of pandas data frames detailing the repayment simulation
                   data for each loan.
        """
        # Set minimum monthly payment
        for loan in self.loans:
            loan.mp  = loan.min_mp
            loan.dmp = 0.0

        heap = MinHeap(initial=self.loans, key=key)

        while len(heap) > 0:
            # Get loan with lowest key
            top_loan = heap[0]

            # If top_loan will be repayed, find new one
            if top_loan.mp >= top_loan.balance:
                heap.replace(top_loan, key=float("inf"))
                top_loan = heap[0]

            # Simulate payment of high-key other loans
            other_payment = sum( loan.simulate_payment() for loan in heap[1:] )
            top_payment = self.max_payment - other_payment

            # Set payment for top loan
            top_loan.mp  = top_payment
            top_loan.dmp = top_payment
            top_loan.simulate_payment()

            # Remove payed off loans and re-sort
            heap.filter(lambda l: l.balance > 0)

        return [ loan.construct_df() for loan in self.loans ]


    def schedule_max_interest(self):
        """
        Simulates full repayment on all loans. After allocating minimum monthly payments
        to each loan, any surplus is given to the active highest interest loan.

        :return: dfs, a list of pandas data frames detailing the repayment simulation
                   data for each loan.
        """
        # Negate the APR to turn the min heap into a max heap
        return self.schedule_prioritize_one(lambda loan: -loan.apr)


    def schedule_lowest_balance(self):
        """
        Simulates full repayment on all loans. After allocating minimum monthly payments
        to each loan, any surplus is given to the lowest balance loan.

        :return: dfs, a list of pandas data frames detailing the repayment simulation
                   data for each loan.
        """
        return self.schedule_prioritize_one(lambda loan: loan.balance)

