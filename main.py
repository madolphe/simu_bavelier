from teachers.random import RandomTeacher
from students.student import BaseStudent
import numpy as np
import pathlib
from utils.plot import create_gif_from_images

seed = 0
np.random.seed(seed)

if __name__ == '__main__':
    # Some config:
    nb_episodes = 5000
    activity_space_desc = (10, 10)
    expe_name = "base_test"
    path = f"./outputs/{expe_name}"
    subfolders = ["competence", "history", "trajectories"]
    for subfolder in subfolders:
        pathlib.Path(f"{path}/{subfolder}/").mkdir(parents=True, exist_ok=True)

    # Create object:
    student = BaseStudent(activity_space_desc=activity_space_desc,
                          nb_to_neighbour_mastery=3,
                          initial_competence_proportion=0.2,
                          unfeasible_space_proportion=0)
    teacher = RandomTeacher(activity_space_desc)

    # Train loop:
    for activity_nb in range(nb_episodes):
        act = teacher.sample_activity()
        ans = student.get_response(act)
        student.update_competence(act, ans)
        if activity_nb % 100 == 0:
            student.selfie(path, episode_number=activity_nb)
            student.bookmark(episode_number=activity_nb)

    # Plot everything
    create_gif_from_images(f"{path}/competence/", f"{path}/competence/output.gif", duration=500)
    create_gif_from_images(f"{path}/history/", f"{path}/history/output.gif", duration=500)
    student.save_trajectories(path=f"{path}/trajectories/")