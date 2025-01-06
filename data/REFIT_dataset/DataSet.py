"""This code file defines two objects, DataLoader and ActivationGenerator.
The former will be called by the code file "gen data set by activation",
while the latter will be called by the code file "gen activations".
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import _pickle
import random


class DataLoader:
    """A DataLoader class that is used to load data from the training, validation, and test sets."""
    def __init__(self, file_location=[], mixed_file_name=[], appliance_file_name=[], app_name=[], dat_type=None):
        """Parameters
        ----------
        file_location : The folder where the file is read
        mixed_file_name : Mixed-signal filename
        ('house1/channel1.dat', the first channel file of each house is often the total signal file)
        appliance_file_name : The target electrical signal file name
        ('house1/channel2.dat' means the electrical signal corresponding to channel2 of house1, as for
        please refer to the UKDale_building_info, REDD_building_info, REFIT_building_info of 'PreDefine.py' to know more)
        app_name : The name of the appliance
        dat_type : Set a different data type for each variable column that reads the dat data"""
        self.dataloader_name = []  # is then used to preserve the name of the Dattalod
        self.dataloader_name1 = []  # train: tr  test: te  validation: va

        from PreDefine import sample_time, appliance_property, was_appliance
        self._all_app_property = appliance_property
        self.params_appliance = was_appliance  # 'was': windowlength and stride
        self.app_name = app_name

        self.mixed_data = []  # Used to store the total signal of the dataloader
        self.appliance_data = []  # Used to store the target signal of this dataloader
        self.set_data = []  # Save the total signal and the single electrical signal in the set_data columns
        self.activation_data = []  # The output activation signal
        self.dataset = {}  # The final dataset for training (for saving in folder format)

        '''read_origin_data'''
        self.separator = "\s+"  #
        self.col_names = ['time', 'data']  # Read the '.dat' file to give the data a name for each column
        self.usecols = [0, 1]  # Number of columns to read the file (from 0)
        self.file_location = file_location  # The path of the file where the data is read
        self.mixed_file_name = mixed_file_name  # The file name of the total signal data read
        self.appliance_file_name = appliance_file_name  # The file name of the single appliance data read
        if dat_type is None:
            self.types = {'time': str}  # Set a different data type for each variable in the DAT data
        else:
            self.types = dat_type

        '''resample_data'''
        self.resample_seconds = sample_time  # 8s

        '''remove_abnormal_data'''
        self.app_data_name = None
        self.mix_data_name = None

        '''generate_dataset'''
        self.stride = self.params_appliance[self.app_name]['stride']
        self.window_length = self.params_appliance[self.app_name]['windowlength']
        self.batch_num = None
        self.all_activation_set = None

        '''Data entering the network (packaged in batches)'''
        self.input_data = {}

        print('Initialize an object: Dataloader')

    # read
    def read_origin_data(self, file_location, mixed_file_name,
                         appliance_file_name, separator, types, nrows=None):
        '''File reading is performed by entering the name of the target data filename and path file_location,
        and the total signal will be read as well as the target electrical signal DataFrame as output.
        You can change the number of read rows by setting nrows= to facilitate debugging
        (the original data has tens of millions of rows, and the reading is slow)'''

        origin_mixed_data = pd.read_table(
            file_location+'/'+mixed_file_name,
            sep=separator,
            usecols=self.usecols,  # A variable that specifies which number of columns to read
            names=self.col_names,  # If there is no variable name in the original dataset, add a specific variable name
            dtype=types,
            nrows=nrows  # Specifies the number of rows to read, defaulting all rows
            )

        origin_appliance_data = pd.read_table(
            file_location + '/' + appliance_file_name,
            sep=separator,
            usecols=self.usecols,
            names=self.col_names,
            dtype=types,
            nrows=nrows
            )
        return origin_mixed_data, origin_appliance_data

    # resample
    def resample_data(self, index_name=None, mixed_data=None, appliance_data=None, keep_origin=False):
        """By calling the resample_data method, you can use the resample_seconds time property of the object
        (UKDale: 6s, REFIT: 8s, REDD: 3s)
        To resample the total signal and the target electrical signal, the timestamps of the total signal and
        the target electrical signal are first merged, and then the data is resampled based on the merged timestamps.
        In this method, the data is resampled first, and then the data is populated. The final output is resampled and
        padded to remove the data with missing values"""
        assert len(mixed_data[index_name]) > 0, print('Read the data to the object and then resample')
        assert len(appliance_data[index_name]) > 0, print('Read the data to the object and then resample')
        # todo: When debugging, if you want to display the signal, you can use the following code in the terminal
        # plt.figure()
        # plt.plot(mixed_data['time'].astype(int) - 1300000000, mixed_data['data'])
        # plt.plot(appliance_data['time'].astype(int) - 1300000000, appliance_data['data'])
        # plt.title('before resample')
        # plt.legend(['mix', 'app'])

        mixed_data[index_name] = pd.to_datetime(
            pd.to_numeric(mixed_data[index_name]), unit='s')
        appliance_data[index_name] = pd.to_datetime(
            pd.to_numeric(appliance_data[index_name]), unit='s')

        mixed_data.set_index(index_name, inplace=True)
        appliance_data.set_index(index_name, inplace=True)
        mixed_data.columns = ['mix_data']  # Change the column name of the data frame to 'mix_data'
        appliance_data.columns = ['app_data']  # Change the column name of the data frame to 'app_data'
        # todo: When debugging, if you want to display the signal, you can use the following code in the terminal
        # plt.figure()
        # plt.plot(mixed_data['mix_data'])
        # plt.plot(appliance_data['app_data'])
        # plt.title('before resample')
        # plt.legend(['mix', 'app'])

        set_data = mixed_data.join(appliance_data, how='outer').\
            resample(str(self.resample_seconds) + 'S').mean().fillna(method='backfill', limit=1)

        set_data = set_data.dropna()  #
        if not keep_origin:  # If you do not want to keep the object's original data, it is deleted from memory
            del self.appliance_data, self.mixed_data

        set_data.reset_index(inplace=True)
        set_data = set_data.drop(columns='time')  # the index becomes a natural number index starting from 0
        # todo: When debugging, if you want to display the signal, you can use the following code in the terminal
        # plt.figure()
        # plt.plot(set_data['mix_data'])
        # plt.plot(set_data['app_data'])
        # plt.title('after resample')
        # plt.legend(['mix', 'app'])
        return set_data, 'mix_data', 'app_data'

    # remove abnormal
    def remove_abnormal_data(self, set_data, num_remove=2, left_threshold=None, right_threshold=None):
        """Through the left threshold of the outlier and the right threshold of the outlier
        (corresponding to the self._all_app_property) in PreDefine, the outlier is removed, and if a certain point
        jumps severely, That is, if the data point is greater than the left threshold and the right threshold is greater
         than the point data on the right, then the data point is considered to be abnormal.
        The data point will be removed. You can process both mixdata and appdata in the set_data of an object"""
        # if do not specify the thresholds, the built-in properties of the object are used for querying
        if not left_threshold:
            left_threshold = self._all_app_property[self.app_name][3]
        if not right_threshold:
            right_threshold = self._all_app_property[self.app_name][4]

        mix_data = set_data[self.mix_data_name].reset_index(drop=True, inplace=False)
        app_data = set_data[self.app_data_name].reset_index(drop=True, inplace=False)
        del set_data
        # todo: If you are Windows, then leave lines 1 and 3; If you are linux, then leave lines 2 and 4.
        mix_data = np.reshape(mix_data, (-1, 1))  # todo: 1 for windows local
        # mix_data = np.reshape(mix_data.to_numpy(), (-1, 1))  # 2 for linux server
        app_data = np.reshape(app_data, (-1, 1))  # todo: 3 for windows local
        # app_data = np.reshape(app_data.to_numpy(), (-1, 1))  # 4 for linux server
        assert len(mix_data) == len(app_data)
        num = len(app_data)
        #
        _mix_data = []
        _app_data = []
        # Use the remove_abnormal_data multiple times according to the number of times required to use
        # the remove abnormal points as many times as you want
        for times in range(num_remove):
            print(times+1, 'times to remove abnormal for ', 'app: ', self.app_name, 'DataLoader:', self.dataloader_name)
            # Travers the app_data, encounters an anomaly, and replaces with the data from the point before the anomaly
            for i, value in enumerate(app_data):
                if i == 0 or i == num - 1:
                    _mix_data.append(mix_data[i])
                    _app_data.append(value)
                else:
                    if ((value - _app_data[-1] > left_threshold or value - app_data[i + 1] > right_threshold)
                            and i + 2 <= num - 1):
                        if value - app_data[i + 2] > right_threshold and value - _app_data[-1] > left_threshold:
                            # Exception: The point is directly replaced with the data from the previous point
                            _mix_data.append(_mix_data[-1])
                            _app_data.append(_app_data[-1])
                            continue
                        if value - app_data[i + 3] > right_threshold and value - _app_data[-1] > left_threshold:
                            # Exception: The point is directly replaced with the data from the previous point
                            _mix_data.append(_mix_data[-1])
                            _app_data.append(_app_data[-1])
                            continue
                        if value - app_data[i + 4] > right_threshold and value - _app_data[-1] > left_threshold:
                            # Exception: The point is directly replaced with the data from the previous point
                            _mix_data.append(_mix_data[-1])
                            _app_data.append(_app_data[-1])
                            continue
                        else:
                            _mix_data.append(mix_data[i])
                            _app_data.append(value)
                    else:
                        _mix_data.append(mix_data[i])
                        _app_data.append(value)
            # The data is transferred and emptied so that it can be traversed again
            mix_data = _mix_data
            app_data = _app_data
            _mix_data = []
            _app_data = []
            # Travers the mix_data
            for i, value in enumerate(mix_data):
                if i == 0 or i == num - 1:
                    _mix_data.append(value)
                    _app_data.append(app_data[i])
                else:
                    if ((value - _mix_data[-1] > left_threshold or value - mix_data[i + 1] > right_threshold)
                            and i + 2 <= num - 1):
                        if value - mix_data[i + 2] > right_threshold and value - _mix_data[-1] > left_threshold:
                            # Exception: The point is directly replaced with the data from the previous point
                            _mix_data.append(_mix_data[-1])
                            _app_data.append(_app_data[-1])
                            continue
                        if value - mix_data[i + 3] > right_threshold and value - _mix_data[-1] > left_threshold:
                            # Exception: The point is directly replaced with the data from the previous point
                            _mix_data.append(_mix_data[-1])
                            _app_data.append(_app_data[-1])
                            continue
                        if value - mix_data[i + 4] > right_threshold and value - _mix_data[-1] > left_threshold:
                            # Exception: The point is directly replaced with the data from the previous point
                            _mix_data.append(_mix_data[-1])
                            _app_data.append(_app_data[-1])
                            continue
                        else:
                            _mix_data.append(value)
                            _app_data.append(app_data[i])
                    else:
                        _mix_data.append(value)
                        _app_data.append(app_data[i])
            mix_data = _mix_data
            app_data = _app_data
            _mix_data = []
            _app_data = []
        set_data = np.concatenate([np.array(mix_data), np.array(app_data)], axis=1)
        return set_data

    def remove_big_abnormal(self,flag=0,limitation=250):
        """Remove the signal whose signal value is greater than the limit in the target electrical signal,
        and the removal method is to replace the difference in the forward direction, only fridge use this
                for signal before:   100, 101, 100, 500, 501, 502, 100, 100
                then after:          101, 100, 100, 101, 102, 100, 100, 100"""
        if self.dataloader_name == 'house_15' or 'house_20': limitation = 150
        else: limitation = 250
        for i in range(len(self.set_data)):
            if self.set_data[i, 1] >= limitation:
                if flag == 0:
                    temp = self.set_data[i, 1] - self.set_data[i - 1, 1]
                    self.set_data[i, 1] = max(self.set_data[i - 1, 1], 0)
                if flag == 1:
                    self.set_data[i, 1] = max(self.set_data[i, 1] - temp, 0)
                flag = 1
            else:
                flag = 0
        # self.set_data[:, 0]+=self.set_data[:, 1]
        return self.set_data

    # Generate dataset data, in which sythetic_ratio part of the data is generated data,
    # and internal functions will be called _synthetic_data generate it
    def generate_dataset(self, data, sythetic_ratio=0, work_probability=0.8, batch_num=None):
        """"Split the raw data by windowlength and stride"""
        if not batch_num:  # This is the total number of copies of the cutting signal
            batch_num = int(np.floor((len(data) - self.window_length) / self.stride))
        stride = self.stride
        window_length = self.window_length
        sythetic_num = sythetic_ratio*batch_num
        mix_set = []
        app_set = []

        for i in range(int(batch_num)):
            temp_mix = data[i * stride:i * stride + window_length, 0]
            temp_app = data[i * stride:i * stride + window_length, 1]
            mix_set.append(temp_mix)
            app_set.append(temp_app)

        for i in range(int(sythetic_num)):
            activation_data = self.all_activation_set
            temp_sythetic = self._synthetic_data(window_length, activation_data, work_probability)
            mix_set.append(temp_sythetic[:, 0])
            app_set.append(temp_sythetic[:, 1])

        mix_set = np.array(mix_set, dtype=float)
        app_set = np.array(app_set, dtype=float)
        temp = {self.app_name: [mix_set, app_set]}
        return temp, batch_num

    # The dataset generation function, an internal function, has been called in the generate dataset
    def _synthetic_data(self, window_length, activation_data, work_prob=0.8):
        """According to the input window length window_length and the hypothetical operating probability of
        the electrical appliance work_prob, the activation_data is used to generate the synthesized
        target electrical data and total signal data"""
        agg = np.zeros(window_length)  # The aggregate signal generated
        source = self.dataloader_name1  # tr te va
        activation_data = activation_data[source]  # Select as the source for the subsequent generation of data
        # Go through all the appliances
        for name in activation_data:
            # The probability of work_prob is randomly taken from the set of target electrical activation signals
            # to generate a training signal
            if np.random.rand() < work_prob:
                data_num = len(activation_data[name])
                temp_ind = np.random.randint(0, data_num)
                temp_data = activation_data[name][temp_ind]
                if len(temp_data) >= window_length:  # If the length is greater than window length,  -5 points
                    temp_data = temp_data[0:window_length - 5]
                # Randomly punch Temple and insert him into a window that looks like Windorunth
                temp_length = window_length - len(temp_data)
                n = np.random.randint(0, temp_length)
                # In the rest of the places, the signal value is zeroed
                temp_data = np.append(np.append(np.zeros(n), temp_data), np.zeros(temp_length - n))
            else:
                temp_data = np.zeros(window_length)
            agg = agg+temp_data+10
            # If the name corresponds, then take out the generated signal of the appliance as the output target
            if name == self.app_name:
                tar = temp_data
        agg = np.reshape(agg, (-1, 1))
        tar = np.reshape(tar, (-1, 1))
        out = np.concatenate([agg, tar], axis=1)
        return out

    # Based on the batch size, the data generated by the previous generate_dataset is used to generate further data
    def generate_data_batch(self, data_set_name, batch_size, all_ap_num, soft_value=30):
        """data_set_name: Index the activation signals in the activation signal set and put them into each batch
        batch_size: Used to extract batch_size large and small bars from the set of activated signals at a time
        (If the size of the batch_size is larger than the length of the activation set, then the activation signal is
        repeatedly put into the batch)
        all_ap_num: Count the number of repetitions of the second type of signal,
        soft_value: Extend the length of the activation signal that is too short (fill with 0)"""
        data_set = self.dataset
        all_acti_data = self.all_activation_set
        input_data = self.input_data
        batch_num = int(np.floor(len(data_set[self.app_name][0]) / batch_size))
        for key, value in data_set.items():
            # batch_num = int(np.floor(len(value[0]) / batch_size))
            act_num = len(all_acti_data[data_set_name][key])
            for q in range(act_num):
                all_acti_data[data_set_name][key][q] = all_acti_data[data_set_name][key][q].astype(float)
                if len(all_acti_data[data_set_name][key][q]) < soft_value:  # todo: The soft value may be set as needed
                    all_acti_data[data_set_name][key][q] = (
                        np.concatenate([all_acti_data[data_set_name][key][q], np.zeros(30 - len(all_acti_data[data_set_name][key][q]))]))
            y_tot = []
            for i in range(batch_num):
                y_bat = value[0][i * batch_size:(i + 1) * batch_size, :]
                tar = value[1][i * batch_size:(i + 1) * batch_size, :]
                if batch_size <= act_num:  # In order to prevent too little activation signal
                    act_ind = random.sample(range(act_num), batch_size)
                else:  # An activation signal is reused
                    act_ind = []
                    for i1 in range(act_num):
                        act_ind.append(int(i1))
                    for i1 in range(batch_size-act_num):
                        act_ind.append(random.randint(0, act_num)-1)
                # act_ind = np.random.randint(0, act_num, batch_size)
                act_len = [len(all_acti_data[data_set_name][key][ind]) for ind in act_ind]
                maxlen = max(act_len)
                act = np.array([np.concatenate([all_acti_data[data_set_name][key][ind],
                                                np.zeros(maxlen - len(all_acti_data[data_set_name][key][ind]))])
                                for ind in act_ind])
                y_tot.append([y_bat, tar, act, act_len])
            input_data[key] = y_tot

        for key in all_acti_data[data_set_name]:
            if key == self.app_name:  # the target electrical signals are skipped
                continue
            if data_set_name == 'te':  # The test set does not need to generate a second type of signal
                continue
            act_num = len(all_acti_data[data_set_name][key])
            for q in range(act_num):
                all_acti_data[data_set_name][key][q] = all_acti_data[data_set_name][key][q].astype(float)
                if len(all_acti_data[data_set_name][key][q]) < soft_value:  # todo: soft value值或许可以按需设置
                    all_acti_data[data_set_name][key][q] = np.concatenate([all_acti_data[data_set_name][key][q],
                                                        np.zeros(30 - len(all_acti_data[data_set_name][key][q]))])
            y_tot = []
            for i in range(int(batch_num/(all_ap_num-1))):
                y_bat = value[0][i * batch_size:(i + 1) * batch_size, :]
                tar = value[1][i * batch_size:(i + 1) * batch_size, :]
                act_ind = np.random.randint(0, act_num, batch_size)
                act_len = [len(all_acti_data[data_set_name][key][ind]) for ind in act_ind]
                maxlen = max(act_len)
                act = np.array([np.concatenate([all_acti_data[data_set_name][key][ind],
                                                np.zeros(maxlen - len(all_acti_data[data_set_name][key][ind]))])
                                for ind in act_ind])
                y_tot.append([act, act_len])
            input_data[key] = y_tot
        return input_data

    # Implement the transfer of datasets between objects
    def transfer_data(self, tar_obj, percent):
        """Transfer the dataset percentage (percent) data of self to the target object"""
        temp_set_tar = tar_obj.dataset[tar_obj.app_name]
        temp_len = int(len(temp_set_tar[0])*percent)
        temp_set = self.dataset[self.app_name][0]
        self.dataset[self.app_name][0] = np.append(temp_set,temp_set_tar[-temp_len:][0],axis=0)
        tar_obj.dataset[tar_obj.app_name][0] = np.delete(temp_set_tar[0], np.s_[-temp_len:], axis=0)
        temp_set = self.dataset[self.app_name][1]
        self.dataset[self.app_name][1] = np.append(temp_set,temp_set_tar[-temp_len:][1],axis=0)
        tar_obj.dataset[tar_obj.app_name][1] = np.delete(temp_set_tar[1], np.s_[-temp_len:], axis=0)
        return self.dataset


class ActivationGenerator(DataLoader):
    """Takes the electrical activation signals for all houses and consolidates them into a dictionary,
    saves them to the property activation_data, and inherits from the parent DataLoader"""

    def read_app_data(self, file_location=None, appliance_file_name=None, separator=None, types=None, nrows=None):
        """Only the signals of the target appliance are read and used for the extraction and generation of the activation signal set"""
        origin_appliance_data = pd.read_table(
            file_location + '/' + appliance_file_name,
            sep=separator,
            usecols=self.usecols,  # Specify which columns of variables to read
            names=self.col_names,  # If no variable name in original dataset, add specific variable names to the data
            dtype=types,
            nrows=nrows
            )
        return origin_appliance_data

    def resample_app_data(self, index_name, appliance_data):
        """By calling the resample_data method, you can use the resample_seconds time property
        of the object (UKDale: 6s, REFIT: 8s, REDD: 3s)
        Only the target electrical signal is resampled (the data is resampled first, and then the data is populated.
        The final output is resampled and padded to remove the missing values of the data)"""

        assert len(appliance_data[index_name]) > 0, print('Read the data to the object and then resample')

        appliance_data[index_name] = pd.to_datetime(
            pd.to_numeric(appliance_data[index_name]), unit='s')
        appliance_data.set_index(index_name, inplace=True)
        appliance_data.columns = ['app_data']  # Change the column name of the data frame to 'app_data'
        appliance_data = (appliance_data.resample(str(self.resample_seconds) + 'S').
                          mean().fillna(method='backfill', limit=1))
        appliance_data = appliance_data.dropna()  # The dropna() function removes all rows that contain missing values
        return appliance_data, 'app_data'

    def remove_abnormal_data(self, appliance_data, num_remove=2, left_threshold=None, right_threshold=None):
        """Through the left threshold of the outlier and the right threshold of the outlier
        (corresponding to the self._all_app_property) in PreDefine, the outlier is removed,
        and if a certain point jumps severely, That is, if the data point is greater than the left threshold and
        the right threshold is greater than the point data on the right,
           then the data point is considered to be abnormal. The data point will be removed."""
        # If you do not specify the thresholds, the built-in properties of the object are used for querying
        if not left_threshold:
            left_threshold = self._all_app_property[self.app_name][3]
        if not right_threshold:
            right_threshold = self._all_app_property[self.app_name][4]
        app_data = appliance_data[self.app_data_name].reset_index(drop=True, inplace=False)
        del appliance_data

        num = len(app_data)
        _app_data = []
        for times in range(num_remove):
            print(times+1, 'times to remove abnormal for ', 'app: ', self.app_name, 'DataLoader:', self.dataloader_name)
            # delete the anomaly of the app_data, and replace it with the data of the previous point
            for i, value in enumerate(app_data):
                if i == 0 or i == num - 1:
                    _app_data.append(value)
                else:
                    if ((value - _app_data[-1] > left_threshold or value - app_data[i + 1] > right_threshold)
                            and i + 2 <= num - 1):
                        if value - app_data[i + 2] > right_threshold and value - _app_data[-1] > left_threshold:
                            _app_data.append(_app_data[-1])
                            continue
                        if value - app_data[i + 3] > right_threshold and value - _app_data[-1] > left_threshold:
                            _app_data.append(_app_data[-1])
                            continue
                        if value - app_data[i + 4] > right_threshold and value - _app_data[-1] > left_threshold:
                            _app_data.append(_app_data[-1])
                            continue
                        else:
                            _app_data.append(value)
                    else:
                        _app_data.append(value)
            # Make sure to reset the _app_data each time and assign a value to the app data
            app_data = _app_data
            _app_data = []
        app_data = np.array(app_data)
        return app_data

    def remove_big_abnormal(self, flag=0, limitation=250):
        """Remove the signal in the fridge electrical signal with a signal value greater than the limitation,
        and the removal method is to replace the difference in the forward direction
        before   100, 101, 100, 500, 501, 502, 100, 100
        after    100, 101, 100, 100, 101, 102, 100, 100"""
        if self.dataloader_name == 'house_15' or 'house_20': limitation = 150
        else: limitation = 250

        for i in range(len(self.appliance_data)):
            if self.appliance_data[i] >= limitation:  # 每个数据集的指标不一样，请按需设置，此指标对应REFIT-house5的Fridge
                if flag == 0:
                    temp = max(0, self.appliance_data[i] - self.appliance_data[i - 1])
                    self.appliance_data[i] = self.appliance_data[i - 1]
                if flag == 1:
                    self.appliance_data[i] = max(0, self.appliance_data[i] - temp)
                flag = 1
            else:
                flag = 0
        return self.appliance_data

    def get_activations(self, chunk, name=None, percent=1,
                        border=2, sample_seconds=None):
        """Extraction of electrical activation signals, adapted from the toolkit NILMTK
        Most electrical appliances tend to account for a small part of the total total time,
        and this function is used to obtain the signal status of electrical appliances when they are working
        Parameters
        ----------
        name : The name of the appliance
        chunk : pd. Series
        border : int
            The larger the number of rows to include before and after an activation is detected,
            the shorter the percentage of the actual work in the activation signal
        sample_seconds :
            Sampling time
        min_off_duration : int, Defaults to 0.
            If min_off_duration > 0, the number of seconds of 'off' cycle power consumption less than
            the Min_off_duration threshold is ignored
            For example, a washing machine may be temporarily unpowered while clothes are soaked.
        min_on_duration : int, Defaults to 0.
            Any activations that last less than min_on_duration will be ignored
        on_power_threshold : int or float
            Watts
        returns : list of pd. Series.
            Each series contains an activation signal
        """
        chunk1 = np.expand_dims(chunk, axis=1)
        chunk1 = list(chunk1)
        chunk1 = chunk1[0:int(percent * len(chunk1))]  # percentage data is taken as the source of activation signal
        name = self.app_name
        chunk = pd.Series(chunk1)
        if not sample_seconds:
            sample_seconds = self.resample_seconds
        min_off_duration = self._all_app_property[name][0]
        min_on_duration = self._all_app_property[name][1]
        on_power_threshold = self._all_app_property[name][2]

        min_off_duration = np.ceil(min_off_duration / sample_seconds)
        min_on_duration = np.ceil(min_on_duration / sample_seconds)
        when_on = chunk >= on_power_threshold

        # Find state changes
        state_changes = when_on.astype(np.int8).diff()
        del when_on
        switch_on_events = np.where(state_changes == 1)[0]
        switch_off_events = np.where(state_changes == -1)[0]
        del state_changes

        if len(switch_on_events) == 0 or len(switch_off_events) == 0:
            return []

        # Make sure events align
        if switch_off_events[0] < switch_on_events[0]:
            switch_off_events = switch_off_events[1:]
            if len(switch_off_events) == 0:
                return []
        if switch_on_events[-1] > switch_off_events[-1]:
            switch_on_events = switch_on_events[:-1]
            if len(switch_on_events) == 0:
                return []
        assert len(switch_on_events) == len(switch_off_events)

        # Smooth over off-durations less than min_off_duration
        if min_off_duration > 0:
            off_durations = (chunk.index[switch_on_events[1:]].values -
                             chunk.index[switch_off_events[:-1]].values)
            above_threshold_off_durations = np.where(off_durations >= min_off_duration)[0]
            # Now remove off_events and on_events
            switch_off_events = switch_off_events[np.concatenate([above_threshold_off_durations,
                                                                  [len(switch_off_events) - 1]])]
            switch_on_events = switch_on_events[np.concatenate([[0], above_threshold_off_durations + 1])]
        assert len(switch_on_events) == len(switch_off_events)

        activations = []
        for on, off in zip(switch_on_events, switch_off_events):
            duration = (chunk.index[off] - chunk.index[on])
            if duration < min_on_duration:
                continue
            on -= 1 + border
            if on < 0:
                on = 0
            off += border
            activation = chunk.iloc[on:off]
            # throw away any activation with any NaN values
            if not activation.isnull().values.any():
                activation = np.array(activation)
                activations.append(activation)
        # todo: draw and see if you want
        # for i in range(5):
        #     plt.plot(activations[i])
        #     plt.title(self.app_name + ' ' + self.dataloader_name)
        self.activation_data = activations
        return activations


print('import DataLoader finished\n')
