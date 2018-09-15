import numpy as np
from gym.utils import seeding

class RandGen:
    """
    Random value generator
    """

    def __init__(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)

    def int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.randint(low, high)

    def float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.randint(0, 2) == 0)

    def elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self.int(0, len(lst))
        return lst[idx]

    def color(self):
        """
        Pick a random color name
        """

        from .miniworld import COLOR_NAMES
        return self.elem(COLOR_NAMES)

    def subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self.elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out
