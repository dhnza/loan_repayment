import pandas as pd
import heapq


def print_schedules(lst):
    """"Compresses a list of repayment schedules into a single data frame, which is then printed."""
    dfs = [ df.set_index('Date') for df in lst ]
    df = pd.concat(dfs, axis=1, sort=False).reset_index().rename(columns={'index':'Date'}).drop('Principal',axis=1)
    df['Total'] = df.filter(axis=1,regex='Payment').sum(axis=1)  # compute total payment
    with pd.option_context('display.max_rows', None,'display.float_format','{:,.2f}'.format):
        print(df)


def isclose(a,b,tol):
    """Retruns true if a and b are within tol of each other, false otherwise."""
    return abs(a-b) <= tol


class MinHeap(object):
    """
    A simple minimum heap with a customizable sort key.
    Source: https://stackoverflow.com/questions/8875706/heapq-with-custom-compare-predicate/8875823
    """
    def __init__(self, initial=None, key=lambda x:x):
        self.key = key
        if initial:
            self._data = [(key(item), item) for item in initial]
            heapq.heapify(self._data)
        else:
            self._data = []

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return (pair[1] for pair in self._data[key])
        else:
            return self._data[key][1]

    def __str__(self):
        return str(self._data)

    def push(self, item, key=None):
        """
        Push item to heap. Default key can be overriden with 'key'.
        """
        if key is not None:
            heapq.heappush(self._data, (key, item))
        else:
            heapq.heappush(self._data, (self.key(item), item))

    def pop(self):
        return heapq.heappop(self._data)[1]

    def replace(self, item, key=None):
        """
        Pop and return the smallest item from the heap, and also push the new item.
        Default key can be overriden with 'key'.
        """
        if key is not None:
            heapq.heapreplace(self._data, (key, item))
        else:
            heapq.heapreplace(self._data, (self.key(item), item))

    def filter(self, pred):
        "Removes elements from the heap that don't match pred in O(n) time"
        self._data = list(filter(lambda p: pred(p[1]), self._data))
        heapq.heapify(self._data)
