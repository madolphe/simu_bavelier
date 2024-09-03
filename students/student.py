import numpy as np
import math
from utils.plot import plot_hypercube_subplots, plot_trajectory, create_gif_from_images
import pathlib
import copy


class BaseStudent:
    def __init__(self, activity_space_desc, nb_to_neighbour_mastery: int, initial_competence_proportion: float,
                 unfeasible_space_proportion: float, unfeasible_space_coordinates=[],
                 initial_competence_space_coordinates=[],
                 exp_name="base_test",
                 flashback=False,
                 proba_success=1,
                 flashback_save=300,
                 flashback_refresh=3000
                 ):
        if type(activity_space_desc) is list:
            activity_space_desc = tuple(activity_space_desc)
        self.activity_space_desc = activity_space_desc
        self.competence_space = np.zeros(activity_space_desc)
        self.history_space = np.zeros(activity_space_desc)
        self.nb_to_neighbour_mastery = nb_to_neighbour_mastery
        self.unfeasible_space_coordinates = unfeasible_space_coordinates
        self.initial_competence_space_coordinates = initial_competence_space_coordinates
        self.initial_competence_proportion = initial_competence_proportion
        self.unfeasible_space_proportion = unfeasible_space_proportion
        self.add_initial_competence()
        self.add_unfeasible_competence()
        self.bk = {
            "bk_index": [],
            "mastery_level": [],
            "history": []
        }
        self.save_path_competence = f"./outputs/{exp_name}/competence/{self.initial_competence_proportion}-{self.unfeasible_space_proportion}-{self.nb_to_neighbour_mastery}"
        self.save_path_history = f"./outputs/{exp_name}/history/{self.initial_competence_proportion}-{self.unfeasible_space_proportion}-{self.nb_to_neighbour_mastery}"
        self.save_path_trajectory = f"./outputs/{exp_name}/trajectory/{self.initial_competence_proportion}-{self.unfeasible_space_proportion}-{self.nb_to_neighbour_mastery}"
        pathlib.Path(self.save_path_competence).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.save_path_history).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.save_path_trajectory).mkdir(parents=True, exist_ok=True)
        self.flashback = flashback
        self.proba_success = proba_success
        self.flashback_save = flashback_save
        self.flashback_refresh = flashback_refresh

    def add_initial_competence(self):
        max_lvl_per_dim = self.find_proportion_cube_limits(self.initial_competence_proportion)
        slices = tuple(slice(0, max_lvl_per_dim) for _ in self.activity_space_desc)
        self.competence_space[slices] = 1
        # Just for correctness, let's fix the new initial competence proportion:
        self.initial_competence_proportion = np.power(max_lvl_per_dim, len(self.activity_space_desc)) / math.prod(
            self.activity_space_desc)

        # An other way to define an unfeasible space is to provide a range of values
        if len(self.initial_competence_space_coordinates) > 0:
            slices = tuple(slice(start, stop) for start, stop in self.initial_competence_space_coordinates)
            # Use the slices to index the array
            self.competence_space[slices] = 1

    def add_unfeasible_competence(self):
        max_lvl_per_dim = self.find_proportion_cube_limits(self.unfeasible_space_proportion)
        self.nb_unfeasible_activities = np.power(max_lvl_per_dim, len(self.activity_space_desc))
        slices = tuple(slice(max - max_lvl_per_dim, max) for max in self.activity_space_desc)
        self.competence_space[slices] = -1
        # Just for correctness, let's fix the new final competence proportion:
        self.unfeasible_space_proportion = np.power(max_lvl_per_dim, len(self.activity_space_desc)) / math.prod(
            self.activity_space_desc)

        # An other way to define an unfeasible space is to provide a range of values
        if len(self.unfeasible_space_coordinates) > 0:
            slices = tuple(slice(start, stop) for start, stop in self.unfeasible_space_coordinates)
            # Use the slices to index the array
            self.competence_space[slices] = -1

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
        self.bk['history'].append(copy.deepcopy(activity))
        if self.competence_space[tuple(activity)] == 1:
            if np.random.random() < self.proba_success:
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
        return np.sum((self.competence_space == 1) & (self.history_space > 1))

    def bookmark(self, episode_number):
        self.bk['bk_index'].append(episode_number)
        self.bk['mastery_level'].append(self.get_mastery())
        if self.flashback and episode_number == self.flashback_save:
            self.bk['competence_episode_flasback'] = copy.deepcopy(self.competence_space)
        elif self.flashback and episode_number == self.flashback_refresh:
            self.competence_space = self.bk['competence_episode_flasback']

    def selfie(self, episode_number):
        plot_hypercube_subplots(self.competence_space, self.save_path_competence, name=f"{episode_number}",
                                episode_number=episode_number, vmin=-1, vmax=1)
        plot_hypercube_subplots(self.history_space, self.save_path_history, name=f"{episode_number}",
                                episode_number=episode_number, vmin=0, vmax=100)

    def create_gif(self):
        create_gif_from_images(self.save_path_competence, f"{self.save_path_competence}/output.gif", duration=500)
        create_gif_from_images(self.save_path_history, f"{self.save_path_history}/output.gif", duration=500)

    def save_trajectories(self):
        plot_trajectory(self.bk['bk_index'], self.bk['mastery_level'], path=self.save_path_trajectory, name='mastery')
