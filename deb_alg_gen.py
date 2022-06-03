from algorithmic_generators import generators
from algorithmic_curriculum import Curriculum

data_generators = generators["add"]
print(data_generators.get_batch(5, 8))
