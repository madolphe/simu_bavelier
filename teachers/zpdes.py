import teachers.kidlearn.kidlearn_lib as k_lib
from teachers.kidlearn.kidlearn_lib import functions as func
from teachers.random_teacher import RandomTeacher


class Zpdes2D(RandomTeacher):
    def __init__(self, activity_space_desc, config_folder_path, file_name):
        super().__init__(activity_space_desc)
        zpdes_params = func.load_json(file_name=file_name, dir_path=config_folder_path)
        self.teacher = k_lib.seq_manager.ZpdesHssbg(zpdes_params)
        self.main_levels = {i: f'nb{i+2}' for i in range(0, 8)}

    def sample_activity(self):
        act = self.teacher.sample()
        return [act['MAIN'][0], act[list(act.keys())[1]][0]]

    def update(self, act, ans):
        index_main = self.main_levels[act[0]]
        print(ans)
        ans = 1 if ans else 0
        print({'MAIN': [act[0]], index_main: [act[1]]}, ans)
        self.teacher.update({'MAIN': [act[0]], index_main: [act[1], act[1], act[1], act[1]]}, ans)


class Zpdes(RandomTeacher):
    def __init__(self, activity_space_desc, config_folder_path, file_name):
        super().__init__(activity_space_desc)
        zpdes_params = func.load_json(file_name=file_name, dir_path=config_folder_path)
        self.teacher = k_lib.seq_manager.ZpdesHssbg(zpdes_params)
        self.main_levels = {i: f'nb{i+1}' for i in range(0, 7)}

    def sample_activity(self):
        act = self.teacher.sample()
        return [act['MAIN'][0], *act[list(act.keys())[1]][:3]]

    def update(self, act, ans):
        index_main = self.main_levels[act[0]]
        ans = 1 if ans else 0
        # Last dim is independant: act[1]
        # Avoid updating the graph
        self.teacher.update({'MAIN': [act[0]], index_main: [act[1], act[2], act[3], act[1]]}, ans)

