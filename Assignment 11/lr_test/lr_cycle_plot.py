import matplotlib.pyplot as plt
import numpy as np
from math import floor

class LRCyclePlot(object):
	"""docstring for LRCyclePlot"""
	def __init__(self, arg):
		super(LRCyclePlot, self).__init__()
		self.iterations = arg.get("iterations", 300)
		self.lr_max = arg.get("lr_max", 2)
		self.lr_min = self.lr_max / 10
		self.step_size = arg.get("step_size", 50)
		self.cycle_size = self.step_size * 2

	def _calc_lr(self, curr_iter):
		cycle = floor(1 + curr_iter/self.cycle_size)
		x = abs(curr_iter/self.step_size - (2 * cycle) + 1)
		lr_t = self.lr_min + (self.lr_max - self.lr_min)*(1 - x)
		return lr_t

	def __call__(self, fname):
		lr_trend = []
		for x in range(self.iterations):
			lr_trend.append(self._calc_lr(x))

		plt.plot(lr_trend)
		plt.title(f'Learning Rate Schedule')
		plt.xlabel('Iterations')
		plt.ylabel('Learning Rate')

		plt.show()
		plt.savefig(fname)

if __name__ == '__main__':
	lrplt = LRCyclePlot({})
	lrplt("lr_cycle_plot_test.png")
