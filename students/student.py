import numpy as np
import math
from utils.plot import plot_hypercube_subplots, plot_trajectory


class BaseStudent:
    def __init__(self, activity_space_desc: tuple, nb_to_neighbour_mastery: int, initial_competence_proportion: float,
                 unfeasible_space_proportion: float):
        self.activity_space_desc = activity_space_desc
        self.competence_space = np.zeros(activity_space_desc)
        self.history_space = np.zeros(activity_space_desc)
        self.nb_to_neighbour_mastery = nb_to_neighbour_mastery
        self.initial_competence_proportion = initial_competence_proportion
        self.unfeasible_space_proportion = unfeasible_space_proportion
        self.add_initial_competence()
        self.add_unfeasible_competence()
        self.bk = {
            "bk_index": [],
            "mastery_level": [],
        }

    def add_initial_competence(self):
        max_lvl_per_dim = self.find_proportion_cube_limits(self.initial_competence_proportion)
        slices = tuple(slice(0, max_lvl_per_dim) for _ in self.activity_space_desc)
        self.competence_space[slices] = 1
        # Just for correctness, let's fix the new initial competence proportion:
        self.initial_competence_proportion = np.power(max_lvl_per_dim, len(self.activity_space_desc)) / math.prod(
            self.activity_space_desc)

    def add_unfeasible_competence(self):
        max_lvl_per_dim = self.find_proportion_cube_limits(self.unfeasible_space_proportion)
        self.nb_unfeasible_activities = np.power(max_lvl_per_dim, len(self.activity_space_desc))
        slices = tuple(slice(max - max_lvl_per_dim, max) for max in self.activity_space_desc)
        self.competence_space[slices] = -1
        # Just for correctness, let's fix the new final competence proportion:
        self.unfeasible_space_proportion = np.power(max_lvl_per_dim, len(self.activity_space_desc)) / math.prod(
            self.activity_space_desc)

    def find_proportion_cube_limits(self, proportion):
        nb_cubes = math.prod(self.activity_space_desc)
        nb_cubes_in_inital = int(nb_cubes * proportion)
        return int(np.power(nb_cubes_in_inital, 1 / len(self.activity_space_desc)))

    def get_response(self, activity):
        """
        This method has 2 purposes, check that the activity correspond to the needed type (e.g correct shape)
        Get the answer to the activity
        :param activity:
        :return:
        """
        if not self.check_activity(activity):
            raise ValueError("Invalid activity")
        return self.answer(activity)

    def check_activity(self, activity) -> bool:
        if len(activity) != len(self.activity_space_desc):
            return False
        coordinates = tuple(activity)
        return all(0 <= coord < dim for coord, dim in zip(coordinates, self.activity_space_desc))

    def answer(self, activity):
        if self.competence_space[tuple(activity)] == 1:
            return True
        return False

    def update_competence(self, activity, response: bool):
        coordinates = tuple(activity)
        # Check if the activity is already mastered:
        if self.competence_space[coordinates] == 1:
            # Then add sampling to history:
            self.history_space[coordinates] += 1
            # Check for neighbours to add if threshold is met:
            if self.history_space[coordinates] >= self.nb_to_neighbour_mastery:
                # Iterate over neighbours
                nb_dims = len(coordinates)
                shape = self.competence_space.shape
                # For each dim,
                for i in range(nb_dims):
                    # Check you are within bounds:
                    if activity[i] + 1 < shape[i]:
                        # Move the position on this particular dim:
                        activity[i] += 1
                        # Just make sure this is not an unfeasible space:
                        if self.competence_space[tuple(activity)] != -1:
                            self.competence_space[tuple(activity)] = 1
                        activity[i] -= 1

    def get_mastery(self):
        # return np.sum(self.competence_space == 1) / self.competence_space.size
        return np.sum(self.competence_space == 1) / (self.competence_space.size - self.nb_unfeasible_activities)

    def bookmark(self, episode_number):
        self.bk['bk_index'].append(episode_number)
        self.bk['mastery_level'].append(self.get_mastery())

    def selfie(self, path, episode_number):
        plot_hypercube_subplots(self.competence_space, f"{path}/competence/", name=f"{episode_number}",
                                episode_number=episode_number, vmin=-1, vmax=1)
        plot_hypercube_subplots(self.history_space, f"{path}/history/", name=f"{episode_number}",
                                episode_number=episode_number, vmin=0, vmax=100)

    def save_trajectories(self, path):
        plot_trajectory(self.bk['bk_index'], self.bk['mastery_level'], path=path, name='mastery')