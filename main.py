from students.school import School
import numpy as np
import json
from progress.bar import Bar
import shutil

seed = 0
np.random.seed(seed)

if __name__ == '__main__':
    # Some config:
    base_dir = './config'
    expe_file_name = 'expe_config'
    config_expe = json.load(open(f'{base_dir}/{expe_file_name}.json', 'r'))
    school = School(**config_expe)
    # Save the config of the current run:
    shutil.copy(f'{base_dir}/{expe_file_name}.json', f'./outputs/{config_expe["expe_name"]}/{expe_file_name}.json')
    shutil.copy(f'{base_dir}/{config_expe["student_params_path"].split("/")[-1]}', f'./outputs/{config_expe["expe_name"]}/{config_expe["student_params_path"].split("/")[-1]}')
    shutil.copy(f'{base_dir}/{config_expe["teachers_params_path"].split("/")[-1]}', f'./outputs/{config_expe["expe_name"]}/{config_expe["teachers_params_path"].split("/")[-1]}')

    with Bar('Processing...') as bar:
        # Train loop:
        for activity_nb in range(config_expe['nb_episodes']):
            school.teach()
            if activity_nb % 100 == 0:
                school.save(episode_number=activity_nb, take_selfie=False)
                bar.next()

    # Plot and save
    school.get_csv_trajectories()
    school.get_all_trajectories()
    school.plot_difficulty_trajectories()
    # school.create_school_gif()
