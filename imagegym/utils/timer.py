import time


class Timer:
    def __init__(self, tabs=0):
        self.tics = {}
        self.tocs = {}
        self.tabs = ''.join(['\t',]*tabs)

    def tic(self, name):
        self.tics[name] = time.time()

    def toc(self, name):
        self.tocs[name] = time.time()

    def get_interval(self, name):
        return self.tocs[name] - self.tics[name]

    def print(self, name):
        interval = self.get_interval(name)
        print(f"{self.tabs}Time[{name}] = {interval:.4f}")

    def print_ordered(self):
        # return

        names = list(self.tocs.keys())
        intervals_dict = {}
        for name in names:
            intervals_dict[name] = self.get_interval(name)
        sorted_intervals_dict = dict(sorted(intervals_dict.items(), key=lambda item: item[1]))

        print('\n\nPRINT ALL')
        for name in sorted_intervals_dict.keys():
            self.print(name)
