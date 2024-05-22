import numpy as np
import json
from itertools import chain
from itertools import combinations
import pickle
import pandas as pd

class AHP():
    def __str__(self):
        return 'Analytical Hierarchy Process (AHP)'

    def __init__(self, options={}, scale='saaty'):
        if scale == 'saaty':
            self.scale = {'saaty': {1: 'igual importância',
                                    2: 'igual importância',
                                    3: 'fraca importância',
                                    4: 'fraca importância',
                                    5: 'forte importância',
                                    6: 'forte importância',
                                    7: 'muito forte importância',
                                    8: 'muito forte importância',
                                    9: 'extremamente importante'}}
        else:
            self.scale = scale

        self.options = options
        self.data = {
            'input_data': [],
            'classification_matrices': [],
            'classification_matrices_normalized': [],
            'classification_matrices_mean': [],
            'classification_matrices_preference': [],
            'criteria_matrix': [],
            'criteria_matrix_normalized': [],
            'criteria_matrix_mean': []
        }

    def fit(self, *data, verbose=False):
        self.input_data = data
        self.data['classification_matrices'] = self._parse_data(data)[0]
        self.data['criteria_matrix'] = self._parse_data(data)[1]

        self.matrix_pipeline(self.data['classification_matrices'],
                             key='classification_matrices',
                             preference_matrix=True)
        
        self.matrix_pipeline([self.data['criteria_matrix']],
                             key='criteria_matrix')
        
        if verbose:
            self._show_matrices()

    def matrix_pipeline(self, data, key, preference_matrix=True):
        """Perform the transformations along the matrices"""
        for matrix in data:
            normalized_matrix = AHP.matrix_normalizer(matrix)
            self.data[f'{key}_normalized'].append(
                np.array(normalized_matrix)
            )

            mean_criteria_m = AHP.mean_criteria_matrix(normalized_matrix)
            self.data[f'{key}_mean'].append(
                np.array(mean_criteria_m)
            )

        if preference_matrix:
            self.data[f'{key}_preference'] = AHP.preference_matrix(
                [matrix for matrix in self.data[f'{key}_mean']]
            )

    def classificate(self, reshape=False, shape=(-1,1), pretty=False):
        dot_matrix =  np.dot(self.data['classification_matrices_preference'],
                             np.reshape(self.data['criteria_matrix_mean'],(-1,1)))
        if reshape:
            return_matrix = np.reshape(dot_matrix, shape)
        else:
            return_matrix = dot_matrix

        if pretty:
            for i in range(len(return_matrix)):
                print(f'{self.options[i]} =', return_matrix[i])
        else:
            return return_matrix

    def _parse_data(self, data):
        """Parse input data of fit function"""
        classification_matrices = []
        criteria_matrix = []

        if len(data) == 1:
            data = data[0]

        if isinstance(data, dict):
            option_matrices = data['options']
            criteria_m = data['criteria']
            for element in option_matrices:
                classification_matrices.append(
                    AHP._parse_json_matrix(element)
                )

            criteria_matrix = AHP._parse_json_matrix(criteria_m)

        elif (isinstance(data, list) or isinstance(data, tuple)) \
              and isinstance(data[0], list):

            classification_matrices = data[:-1]
            criteria_matrix = data[-1]

        elif (isinstance(data, list) or isinstance(data, tuple)) \
              and isinstance(data[0], dict):

            for element in data[:-1]:
                classification_matrices.append(
                    AHP._parse_json_matrix(element)
                )

            criteria_matrix = AHP._parse_json_matrix(data[-1])

        return np.array(classification_matrices), np.array(criteria_matrix)

    @staticmethod
    def matrix_normalizer(matrix):
        normalized_matrix = matrix.copy()
        normalized_matrix = np.array(normalized_matrix)
        columns_sum = np.sum(matrix, axis=0)
        for i in range(len(matrix)):
            for k in range(len(matrix[i])):
                normalized_matrix[i][k] = (matrix[i][k])/columns_sum[k]
        return normalized_matrix

    @staticmethod
    def mean_criteria_matrix(matrix):
        criteria_means = np.sum(matrix, axis=1)
        mc_matrix = [i/len(matrix[0]) for i in criteria_means]
        return mc_matrix

    @staticmethod
    def preference_matrix(*criteria_matrices):
        stack_elements = [i for matrix in criteria_matrices for i in matrix]
        preference_matrix = np.stack(stack_elements, axis=0)
        preference_matrix = np.transpose(preference_matrix)
        return preference_matrix

    @staticmethod
    def _parse_json_matrix(json_data):
        """json_data to array"""
        unique_options = AHP._get_json_data_unique_options(
            json_data
        )

        lenght = len(unique_options)

        m = np.identity(lenght)
        indexes = list(combinations(range(lenght), 2))

        json_values = list(json_data.values())

        for index, value in zip(indexes, json_values):
            m[index[0]][index[1]] = value
            m[index[1]][index[0]] = value**(-1)
        return m

    @staticmethod
    def _get_json_data_unique_options(json_data):
        """Return options"""
        options = []
        for key in list(json_data.keys()):
            options.append(key)

        unique_options = []
        options = chain(*options)
        for i in options:
            if i not in unique_options:
                unique_options.append(i)
        return unique_options

    @staticmethod
    def save_model(model, out_dir='./', filename='model.sav'):
        pickle.dump(model, open(out_dir + filename, 'wb'))

    def _show_matrices(self, n=90):
        keys = ['classification_matrices',
                'criteria_matrix']
        i = 1
        for key in keys:
            if np.ndim(self.data[key]) < 3:
                matrix = [self.data[key]]
            else:
                matrix = self.data[key]

            print('-'*n + ' ' + key.upper(), end='\n\n')
            for m in matrix:
                print(m, end='\n\n')

            print('-'*int(n/2)+f' STEP {i}: {key}_normalized', end='\n\n')
            print(f'{key}_normalized')
            for m in self.data[f'{key}_normalized']:
                print(m, end='\n\n')

            print('-'*int(n/2)+f' STEP {i+1}: {key}_mean', end='\n\n')
            print(f'{key}_mean')
            for m in self.data[f'{key}_mean']:
                print(m, end='\n\n')

            print('-'*int(n/2)+f' STEP {i+2}: {key}_preference', end='\n\n')
            
            if f'{key}_preference' in list(self.data.keys()):
                print(f'{key}_preference')
                print(self.data[f'{key}_preference'], end='\n\n')
                i += 3
            else:
                i += 2

    @staticmethod
    def pretty_print_matrix(matrix_array, key, title):
        size = len(matrix_array) + 1
        df = pd.DataFrame(data=matrix_array,
                          index=[f'{key}{i}' for i in range(1, size)],
                          columns=[f'{key}{i}' for i in range(1, size)])

        columns = [(title, f'{key}{i}') for i in range(1, size)]

        df.columns = pd.MultiIndex.from_tuples(columns)
        return df