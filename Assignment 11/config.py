import pprint

class ModelConfig(object):

	def __init__(self,):
		super(ModelConfig, self).__init__()
		self.seed = 1
		self.batch_size_cuda = 512
		self.batch_size_cpu = 128	
		self.num_workers = 4
		# Regularization
		self.dropout = 0
		self.l1_decay = 2e-6
		self.l2_decay = 6e-4
		self.lr = 0.001
		self.momentum = 0.9
		self.epochs = 24

	def print_config(self):
		print("Model Parameters:")
		pprint.pprint(vars(self), indent=2)

def test_config():
	args = ModelConfig()
	args.print_config()

if __name__ == '__main__':
	test_config()
