import numpy as np


class RandomTeacher:
    def __init__(self, activity_space_desc):
        self.activity_space_desc = activity_space_desc

    def sample_activity(self):
        return [np.random.randint(0, dim) for dim in self.activity_space_desc]
