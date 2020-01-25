import math
import pandas as pd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from pprint import pformat

from utilities import isclose

class Loan:
    """Simulates loan repayment over time"""


    def __init__(self, principal, apr, min_monthly_payment, term, start_date, interest=0.0, monthly_payment=None, deferment_monthly_payment=0.0, deferment_exit_date=None):
        # Set amounts
        self.init_principal = self.principal = principal
        self.init_interest  = self.interest  = interest
        self.init_balance   = self.balance   = principal + interest

        # Set percentage rates
        self.apr = apr
        self.mpr = apr/12
        self.dpr = apr/365

        # Set monthly payments
        if monthly_payment is None:
            monthly_payment = min_monthly_payment
        self.min_mp   = min_monthly_payment
        self.init_mp  = self.mp  = monthly_payment
        self.init_dmp = self.dmp = deferment_monthly_payment

        # Set dates
        self.term = term
        self.start_date = start_date
        self.end_date = start_date + relativedelta(months=term)
        self.deferment_exit_date = deferment_exit_date
        self.cur_date = date.today()

        # Used to increment a date to the next payment date
        self.next_payment = relativedelta(day=start_date.day, months=1)

        # self.data will store repayment simulation steps
        #   Data order is (date, balance, principal, payment)
        self.data = []
        # Set initial simulation step
        self.data.append( (self.cur_date, self.balance, self.principal, 0) )

        # Validate initialization arguments
        self.validate()


    def __str__(self):
        return pformat(vars(self), indent=4)


    def validate(self):
        """
        Validates the current state of this Loan.

        :raises ValueError: if the monthly payment is less than the minimum for given loan parameters
        :raises ValueError: if the monthly payment is less than the interest accrued over a month
        """
        emp = self.expected_monthly_payment()
        if (self.min_mp < emp) and not isclose(self.min_mp, emp, 0.01):
            raise ValueError("Minimum monthly payment of ${:0,.2f} is less than expected minimum of ${:0,.2f}".format(self.mp, emp))

        ipm = self.mpr*self.principal
        if (self.min_mp < ipm):
            raise ValueError("Minimum monthly payment of ${:0,.2f} is less than interest accrued in a month of ${:0,.2f} ".format(self.mp, ipm))

        if (self.min_mp > self.mp):
            raise ValueError("Minimum monthly payment of ${:0,.2f} exceeds set monthly payment of ${:0,.2f} ".format(self.min_mp, self.mp))


    def expected_monthly_payment(self):
        """
        Calculates the minimum monthly payment for this Loan given the starting principal, APR, and term length.

        :return: mp, the miniumum monthly payment
        """
        return (self.mpr * self.init_principal)/(1-(1+self.mpr)**(-self.term))


    def default_cost(self):
        """
        Calculates the total cost for this loan when paying only the minimum monthly payment for the entire term.

        :return: cost, the default cost of this loan
        """
        return (self.mpr * self.init_principal * self.term)/(1-(1+self.mpr)**(-self.term))


    def months_to_deferment_exit(self, start=date.today()):
        """
        Returns the number of months between the first payment date after start
          and the deferment exit date. Returns 0 if deferment has passed
        """
        next_payment_date = start + self.next_payment
        m = relativedelta(self.deferment_exit_date, next_payment_date).months + 1
        return m if m > 0 else 0


    def months_to_repayment(self):
        """
        Returns the number of months until the loan has been repayed in full, given
          the current principal, APR, and monthly payment.
        """
        return -math.log(1 - self.mpr*self.principal / self.mp) / math.log(1 + self.mpr)


    def simulate_payment(self, n=1):
        """
        Simulates n payments on this loan. The simulation data is added to
          an internal data structure. The internal state of this Loan will
          be modified.

        :param int n: The number of payments to simulate
        :return: payment, the total amount payed over all periods
        """
        total_payment = 0
        for i in range(n):
            prev = self.cur_date
            self.cur_date += self.next_payment
            days = (self.cur_date - prev).days

            # Interest does not accrue during deferment
            if self.cur_date <= self.deferment_exit_date:
                payment = self.dmp
            else:
                payment = self.mp
                self.balance += self.principal*self.dpr*days

            self.balance -= payment
            # Principal is reduced only if all interest is paid off
            self.principal = self.balance if self.balance < self.principal else self.principal

            # Correct over-payment
            if self.balance < 0:
                payment = payment + self.balance
                self.balance = 0
                self.principal = 0

            total_payment += payment
            self.data.append( (self.cur_date, self.balance, self.principal, payment) )

        return total_payment


    def construct_df(self):
        """
        Constructs a pandas data frame detailing the simulation steps taken so far.

        :return: df, the data frame
        """
        return pd.DataFrame(self.data, columns=('Date', 'Balance', 'Principal', 'Payment'))


    # Precondition: self.validate() raises no exceptions
    def simulate_full_repayment(self):
        """
        Continues the simulation until balance is paid in full. The full simulation data
          is then returned as a pandas data frame.

        :return: a 2-tuple containing:
                1. df, the data frame
                2. total_cost, the total cost of this loan
        """
        while self.balance > 0:
            self.simulate_payment()

        df = self.construct_df()
        total_cost = df['Payment'].sum()
        return df, total_cost


    # Precondition: self.validate() raises no exceptions
    def simulate_payments(self, payments):
        """
        Continues the simulation according to the given payments, i.e.
            payment[i] is applied to term i

        :raises ValueError: if payment[i] is less than the minimum payment and
                            the loan has not been paid in full
        """
        for mp, t in zip(payments, range(len(payments))):
            if self.balance <= 0.0:
                break
            if (self.min_mp > mp):
                raise ValueError("Monthly payment of ${:0,.2f} for term {:d} does not cover balance (${:0,.2f}) or minimum monthly payment (${:0,.2f})".format(mp, t, self.balance, self.min_mp))
            self.mp = mp
            self.simulate_payment()
