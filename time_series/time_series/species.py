from collections import defaultdict


class Species:
    def __init__(self):
        self.interactions = defaultdict(lambda: 1.0)
