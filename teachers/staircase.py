import copy
import json
import random

from teachers.random_teacher import RandomTeacher


class Staircase(RandomTeacher):
    def __init__(self, activity_space_desc, initial_position, nb_up=3, initial_step_size=1):
        super().__init__(activity_space_desc)
        self.position = initial_position
        if self.position is None:
            self.position = [0, 0, 0, 0]
        self.success_in_a_row = 0
        self.nb_up = nb_up
        # This can be changed if we want larger step sizes depending on the nb_targets
        self.step_size = initial_step_size
        # This list contains the index of the sublevels that can be randomly sampled
        self.index_sub_levels = [1, 2, 3]
        # This list will be updated (when sampling, a value is removed)
        self.sub_level_history = [1, 2, 3]

        # Useful position in staircase:
        self.max_sub_levels_position = [e-1 for e in activity_space_desc[1:]]
        self.min_sub_levels_position = [0 for _ in activity_space_desc[1:]]

    def sample_activity(self):
        return self.position

    def update(self, act, ans):
        if ans:
            self.success_in_a_row += 1
            if self.check_staircase_update():
                self.move_sublevels_staircase(1 * self.step_size)
        else:
            self.success_in_a_row = 0
            self.move_sublevels_staircase(-1 * self.step_size)

    def check_staircase_update(self):
        if self.success_in_a_row == 3:
            # To restart counting success:
            self.success_in_a_row = 0
            return True
        return False

    def move_sublevels_staircase(self, step):
        # To start with we consider the up case:
        if step > 0:
            if self.position[1:] == self.max_sub_levels_position:
                # Update main level staircase
                self.move_mainlevel_staircase(1)
            else:
                # To ensure the list is not empty (bc all index have been sampled, we check every time):
                if len(self.sub_level_history) == 0:
                    self.sub_level_history = copy.deepcopy(self.index_sub_levels)
                # Otherwise keep the main level constant and move the sublevels
                # First find the value to sample:
                index_sublevel_to_update = random.choice(self.sub_level_history)
                self.position[index_sublevel_to_update] += 1
                # To avoid making this value sampled again we remove it:
                self.sub_level_history.remove(index_sublevel_to_update)
        # Let's look at the down case:
        else:
            if self.position[1:] == self.min_sub_levels_position:
                # Update main level staircase
                self.move_mainlevel_staircase(-1)
            else:
                # To ensure the list is not empty (bc all index have been sampled, we check every time):
                if len(self.sub_level_history) == 3:
                    self.sub_level_history = []
                # Otherwise keep the main level constant and move the sublevels
                # Finding the value to sample is harder because it corresponds to values not in self.sub_level_history
                missing_values = list(set(self.index_sub_levels) - set(self.sub_level_history))
                # Then we sample from this list:
                index_sublevel_to_update = random.choice(missing_values)
                self.position[index_sublevel_to_update] -= 1
                self.sub_level_history.append(index_sublevel_to_update)

    def move_mainlevel_staircase(self, step):
        # First increase the main levels
        if ((0 < self.position[0]) & (step < 0)) or ((self.position[0] < self.activity_space_desc[0]-1) & (step > 0)):
            self.position[0] += step
            # Then move the sublevels staircase on the lowest or highest step:
            self.position[1:] = self.min_sub_levels_position if step > 0 else self.max_sub_levels_position