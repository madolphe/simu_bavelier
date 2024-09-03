import sys
import numpy as np
import pandas as pd
import json
import os
import ast
import matplotlib.pyplot as plt

from pathlib import Path

def params2difranking(data, param_info):
    mainVar = list(param_info['main'].keys())[0]
    mainVarVals = np.array(param_info['main'][mainVar])

    subVars = list(param_info['sub'].keys())

    x = data.loc[:, [mainVar] + subVars].values if isinstance(data, pd.DataFrame) else data

    pmat_dims = np.zeros([len(mainVarVals), len(subVars)])
    sv_dat_all = {}

    for isv, sv in enumerate(subVars):
        sv_dat = np.full(
            [len(param_info['sub'][sv]), max([len(x) if isinstance(x, list) else 1 for x in param_info['sub'][sv]])],
            np.nan)
        for i, j in enumerate(param_info['sub'][sv]):
            sv_dat[i][0:len(j) if isinstance(j, list) else 1] = j
        sv_dat = np.squeeze(sv_dat)
        if sv_dat.ndim == 1:
            sv_dat = np.kron(sv_dat, np.ones([len(mainVarVals), 1]))
        elif (sv_dat.ndim == 2) and (sv_dat.shape[0] != len(mainVarVals)):
            raise Exception(
                'parameter info dict is badly formatted for parameter {}: length of first dimension ({:d}) is not equal to the number of main parameter options ({:d})'.format(
                    sv, sv_dat.shape[0], len(mainVarVals)))
        elif (sv_dat.ndim == 0) or (sv_dat.ndim > 2):
            raise Exception(
                'parameter info dict is badly formatted for parameter {}: the data must either be one or two dimensional'.format(
                    sv))
        sv_dat_all[sv] = sv_dat
        for im in range(len(mainVarVals)):
            pmat_dims[im, isv] = len(sv_dat[im]) - np.isnan(sv_dat[im]).sum()

    lvlgaps = np.nansum(pmat_dims, axis=1) - pmat_dims.shape[1] + 1
    lvloffsets = np.concatenate([[0], np.cumsum(lvlgaps)])
    max_possible_rank = lvlgaps.sum() - 1

    # convert to indices
    # had to initialize the location_seq a little differently because of some bad formatting in the data. Used pandas to_numeric to deal with it
    location_seq = np.zeros_like(x).astype(np.float64)  # x.copy().astype(np.float64)
    for i in range(location_seq.shape[1]):
        location_seq[:, i] = pd.to_numeric(x[:, i], errors='coerce').astype(np.float64)
    dimsize_seq = np.zeros_like(x).astype(int)
    location_seq[:, 0] = np.apply_along_axis(lambda a: np.nanargmin(np.abs(mainVarVals - a[0])), 1,
                                             location_seq[:, [0]])
    dimsize_seq[:, 0] = len(mainVarVals)
    for j, v in enumerate(subVars):
        location_seq[:, j + 1] = np.apply_along_axis(
            lambda a: np.nan if np.isnan(a[1]) else np.nanargmin(np.abs(sv_dat_all[v][int(a[0])] - a[1])), 1,
            location_seq[:, [0, j + 1]])
        dimsize_seq[:, j + 1] = np.apply_along_axis(lambda a: pmat_dims[int(a[0]), j], 1, location_seq[:, [0]])

    # I shouldn't have to do this but for some reason the nans are converted to int which results in a bad value, should further investigate why...
    location_seq[location_seq < 0] = np.nan

    dimsize_seq = dimsize_seq.astype(int)
    difficulty_rank_seq = (np.sum(location_seq[:, 1:], axis=1) + lvloffsets[location_seq[:, 0].astype(int)]).astype(
        np.float64)
    difficulty_seq = difficulty_rank_seq / max_possible_rank  # normalize difficulty

    split_difficulty_seq = location_seq / (dimsize_seq - 1)

    # location_seq = location_seq.astype(np.float64)
    location_seq[np.isnan(location_seq)] = -1

    if np.sum(difficulty_seq < 0) > 0:
        print('WARNING: difficulty below zero')

    return (difficulty_rank_seq, difficulty_seq, location_seq.astype(int), split_difficulty_seq, dimsize_seq)


def get_difficulty(path_to_data, path_to_info):
    usagetxt = 'Requirements: numpy, pandas, json'
    usagetxt += '\nUsage: `python -m difficulty_ranking /path/to/data.csv path/to/parameterInfo.json`'
    usagetxt += '\nExample parameter info json: {"main":{"mainVariableName":[0,1,2,3]}, "sub":{"sub0":[0,1,2,3,4],"sub1":[0,1,2,3,4],"sub2":[0,1,2,3,4]}}'
    usagetxt += '\nNote that all the variable names in the parameter info must be present as column names in the data csv'

    try:
        data = pd.read_csv(path_to_data)
        pathparts = os.path.split(path_to_data)
        basedir = pathparts[0]
        filename = pathparts[1]
    except Exception as ex:
        raise Exception('could not read csv file: {}'.format(ex))

    try:
        pinf = json.load(open(path_to_info))
        assert len(pinf['main'].keys()) == 1, 'There must be exactly ONE main parameter'
        params = list(pinf['main'].keys()) + list(pinf['sub'].keys())
    except Exception as ex:
        raise Exception('could not parse parameter info: {}'.format(ex))

    difrnkseq, difseq, locseq, spldifseq, _ = params2difranking(data[params], pinf)

    df = pd.DataFrame(
        columns=['difficulty_rank', 'difficulty'] + ['{}_{}'.format(clm, v) for clm in ['arrayposition', 'difficulty']
                                                     for v in params])
    df.loc[:, 'difficulty_rank'] = difrnkseq
    df.loc[:, 'difficulty'] = difseq
    for i, p in enumerate(params):
        df.loc[:, 'arrayposition_{}'.format(p)] = locseq[:, i]
        df.loc[:, 'difficulty_{}'.format(p)] = spldifseq[:, i]

    df.to_csv(os.path.join(basedir, filename.split('.')[0] + '_difficulty.csv'))


def reformat_data(data_path, base_dir):
    Path(f"{base_dir}/students_csv").mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_path)
    data = data.drop(columns=['Unnamed: 0'])
    columns = ['mainVariableName', 'sub0', 'sub1', 'sub2']
    for col in data.columns:
        df = pd.DataFrame(columns=columns)
        data[col] = data[col].apply(lambda row: ast.literal_eval(row))
        for i, new_col in enumerate(columns):
            df[new_col] = data[col].apply(lambda row: row[i])
        df.to_csv(f'{base_dir}/students_csv/{col}-trajectories.csv')


def plot_difficulties(basedir, filenames, n=50, plot_all=True, variable="arrayposition_mainVariableName"):
    for filename in filenames:
        data = pd.read_csv(f'{basedir}/students_csv/{filename}_difficulty.csv')
        if plot_all:
            plt.plot([i for i in range(len(data[variable]))], data[variable], label=filename)
        else:
            sliding_mean = np.convolve(data[variable].values, np.ones(n) / n, mode='valid')
            # Compute the sliding window sum of squares using convolution
            squared_mean = np.convolve(data[variable] ** 2, np.ones(n) / n, mode='valid')
            # Calculate the sliding window standard deviation
            sliding_std = np.sqrt(squared_mean - sliding_mean ** 2)
            time_index = np.arange(len(sliding_std))  # Create an index for the x-axis
            # Plot the sliding mean
            plt.plot(time_index, sliding_mean, label=f'Sliding Mean \n {filename}')
            # Plot the standard deviation as shaded error bands
            plt.fill_between(time_index, sliding_mean - sliding_std, sliding_mean + sliding_std, alpha=0.3, label='Standard Deviation')
        plt.xlabel('Time Index')
        plt.ylabel('Difficulty')
        plt.title('Sliding Mean with Standard Deviation')
        # Place the legend outside the plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Adjust the layout to make room for the legend
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f'{basedir}/students_csv/{filename}_difficulty.png')
        plt.close()


if __name__ == "__main__":
    base_dir = "simu_compute_difficulty/"
    data_path = f"{base_dir}/trajectories.csv"
    reformat_data(data_path)
    files_to_compute = ['Zpdes-0-trajectories', 'Staircase-0-trajectories']
    for filename in files_to_compute:
        path_to_data = f"{base_dir}/{filename}.csv"
        path_to_info = f"{base_dir}/pinf.json"
        get_difficulty(path_to_data, path_to_info)
    plot_difficulties(basedir=base_dir, filenames=files_to_compute, variable="difficulty_rank")
