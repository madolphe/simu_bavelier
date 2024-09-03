import copy
import json
import itertools
from students.student import BaseStudent
from teachers.random_teacher import RandomTeacher, RandomTeacherBad
from teachers.zpdes import Zpdes2D, Zpdes
from teachers.staircase import Staircase
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

from utils.compute_difficulty import plot_difficulties, reformat_data, get_difficulty


class School:
    def __init__(self, nb_classrooms: int, student_params_path: str, teachers_params_path: str, expe_name: str,
                 nb_episodes: int):
        self.nb_classrooms = nb_classrooms
        self.classrooms = [copy.deepcopy(ClassRoom(student_params_path, teachers_params_path)) for _ in
                           range(nb_classrooms)]
        save_path = f"./outputs/{expe_name}"
        pathlib.Path(f"{save_path}/").mkdir(parents=True, exist_ok=True)
        self.save_path = save_path
        self.nb_episodes = nb_episodes

    def create_subfolders(self):
        # When the activity space is 2D, create subfolders (otherwise no need)
        subfolders = ["competence", "history"]
        for subfolder in subfolders:
            pathlib.Path(f"{self.save_path}/{subfolder}/").mkdir(parents=True, exist_ok=True)

    def teach(self):
        for classroom_index, classroom in enumerate(self.classrooms):
            # print("Classroom:", classroom_index)
            classroom.teach()

    def save(self, episode_number, take_selfie=False):
        for classroom in self.classrooms:
            classroom.save(episode_number, take_selfie=take_selfie)

    def create_school_gif(self):
        for classroom in self.classrooms:
            classroom.create_gifs()

    def get_all_trajectories(self):
        # Create a place holder for each teacher type:
        classrooms_trajectories = []
        for classroom in self.classrooms:
            all_trajectories, time_index = classroom.get_all_trajectories()
            classrooms_trajectories.append(all_trajectories)
        classrooms_trajectories = np.array(classrooms_trajectories)
        classroom_mean = np.mean(classrooms_trajectories, axis=0)
        teacher_mean, teacher_std = np.mean(classroom_mean, axis=1), np.std(classroom_mean, axis=1)
        for index, teacher_name in enumerate(self.classrooms[0].teacher_names):
            line, = plt.plot(time_index, teacher_mean[index, :], label=teacher_name, linewidth=3.5)
            # plt.fill_between(time_index, teacher_mean[index, :] - teacher_std[index, :],
            #                  teacher_mean[index, :] + teacher_std[index, :], alpha=0.2)
            teacher_color = line.get_color()
            for classroom_index in range(classrooms_trajectories.shape[0]):
                for stu_index in range(classroom_mean.shape[1]):
                    plt.plot(time_index, classrooms_trajectories[classroom_index, index, stu_index, :], c=teacher_color,
                             alpha=0.4, linewidth=0.8)
        # Beautify the plot
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title(
            f'{classrooms_trajectories.shape[0]} classroom(s) \n containing {classroom_mean.shape[1]} student(s).')
        plt.legend(loc='upper right')
        plt.savefig(f"{self.save_path}/trajectories-mastery.png")

    def get_csv_trajectories(self):
        for classroom in self.classrooms:
            classroom.get_csv_trajectories(self.save_path)

    def plot_difficulty_trajectories(self):
        for classroom in self.classrooms:
            classroom.plot_csv_trajectories(self.save_path)


class ClassRoom:
    def __init__(self, student_params_path: str, teachers_params_path: str):
        config = json.load(open(student_params_path, 'r'))
        # Generate all combinations
        keys = config.keys()
        combinations = list(itertools.product(*config.values()))
        # Convert the combinations into a list of dicts (optional, depending on your use case)
        students_params = [dict(zip(keys, combination)) for combination in combinations]
        tmp_students = []
        for student_params in students_params:
            tmp_students.append(BaseStudent(**student_params))
        # Then create as many teacher as student for each teacher type
        config_teachers = json.load(open(teachers_params_path, 'r'))
        self.teachers, self.students = [], []
        for teacher_name, teacher_param in config_teachers.items():
            teacher = self.retrieve_teacher_object_from_name(teacher_name)
            self.teachers.append([copy.deepcopy(teacher(**teacher_param)) for _ in tmp_students])
            self.students.append(copy.deepcopy(tmp_students))
        self.teacher_names = list(config_teachers.keys())

    def teach(self):
        for teacher_type_index in range(len(self.teachers)):
            for stu_teacher_index, (student, teacher) in enumerate(
                    zip(self.students[teacher_type_index], self.teachers[teacher_type_index])):
                act = teacher.sample_activity()
                ans = student.get_response(act)
                # print(f"Teacher {teacher_type_index}, Student {stu_teacher_index}, act: {act}, ans: {ans}")
                student.update_competence(act, ans)
                teacher.update(act, ans)

    def save(self, episode_nb, take_selfie=False):
        for teacher_type_index in range(len(self.teachers)):
            for student, teacher in zip(self.students[teacher_type_index], self.teachers[teacher_type_index]):
                if take_selfie:
                    student.selfie(episode_number=episode_nb)
                student.bookmark(episode_number=episode_nb)

    def create_gifs(self):
        for student_group in self.students:
            for student in student_group:
                student.create_gif()

    def get_all_trajectories(self):
        all_traj = []
        time_index = self.students[0][0].bk['bk_index']
        for student_group in self.students:
            tmp_teacher_all_traj = []
            for student in student_group:
                tmp_teacher_all_traj.append(student.bk['mastery_level'])
            all_traj.append(tmp_teacher_all_traj)
        return all_traj, time_index

    def get_csv_trajectories(self, base_dir):
        csv_dict = {}
        for index_student_group, student_group in enumerate(self.students):
            for index_student, student in enumerate(student_group):
                csv_dict[f"{self.teacher_names[index_student_group]}-{index_student}"] = student.bk['history']
        df = pd.DataFrame.from_dict(csv_dict)
        df.to_csv(f"{base_dir}/trajectories.csv")

    def plot_csv_trajectories(self, base_dir):
        data_path = f"{base_dir}/trajectories.csv"
        reformat_data(data_path, base_dir)
        files_to_compute = [f"{teacher_name}-{i}-trajectories" for i in range(len(self.students[0])) for teacher_name in self.teacher_names]
        for filename in files_to_compute:
            path_to_data = f"{base_dir}/students_csv/{filename}.csv"
            path_to_info = f"./config/pinf.json"
            get_difficulty(path_to_data, path_to_info)
        plot_difficulties(basedir=base_dir, filenames=files_to_compute, variable="difficulty_rank")

    def retrieve_teacher_object_from_name(self, teacher_name):
        if teacher_name == "RandomTeacher":
            return RandomTeacher
        if teacher_name == "RandomTeacher2":
            return RandomTeacherBad
        if teacher_name == "Zpdes2D":
            return Zpdes2D
        if teacher_name == "Zpdes":
            return Zpdes
        if teacher_name == "Staircase":
            return Staircase
        if teacher_name == "RandomStaircaseTeacher":
            pass
