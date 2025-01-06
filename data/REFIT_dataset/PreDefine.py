"""This file contains some predefined variables, Gen Activations, Gen Dataset by Activations,
and Save Dataset by Batch in the same folder
DataSet is called and uses multiple variables in this file"""

ap_name = 'dishwasher'  # app name: 'fridge','kettle','microwave','dishwasher','washingmachine'
data_name = 'REFIT'  # dataset name: 'UKDale','REFIT','REDD'
sample_time = 8  # Time interval for resampling (unit: seconds),UKDale-6,REFIT-8,REDD-3
"""For REDD and UKDale, all of our electrical appliances are the same for training houses and test houses, 
but for REFIT, the current research test houses, It is almost not the same for different appliances, 
so here we need a dictionary to refer to the experiment that β-VAE did in REFIT"""
bVAE_dict = {'fridge': ['house_5', 'house_15', 'house_20'],
             'microwave': ['house_20', 'house_4', 'house_5'],
             'kettle': ['house_20', 'house_2', 'house_5'],
             'dishwasher': ['house_2', 'house_20', 'house_5'],
             'washingmachine':['house_9', 'house_8', 'house_20']}

"""was_appliance('was': window_length_and_stride): The corresponding split window length 
and step size parameters for each appliance"""
was_appliance = {
    'fridge': {'windowlength': 1024, 'stride': 256},
    'kettle': {'windowlength': 128, 'stride': 64},
    'microwave':  {'windowlength': 288, 'stride': 128},
    'dishwasher': {'windowlength': 1536, 'stride': 256},
    'washingmachine': {'windowlength': 2048, 'stride': 512}}

appliance_property = {
    'kettle': [0, 12, 1000, 150, 100],
    'fridge freezer': [12, 60, 50, 30, 25],
    'freezer': [12, 60, 50, 30, 25],
    'fridge': [12, 60, 50, 30, 25],
    'dishwasher': [1800, 1800, 10, 100, 100],
    'microwave': [30, 12, 1000, 150, 100],
    'washer dryer': [160, 1800, 20, 150, 100],
    'washingmachine': [160, 1800, 20, 150, 100],
    'boiler': [0, 0, 10],
    'toaster': [0, 0, 500],
    'hair_dryer': [0, 0, 50],
    'iron': [0, 0, 50],
    'gas oven': [0, 0, 5]}

params_appliance = {  # uk dale
    'kettle': {'windowlength': 128, 'on_power_threshold': 1000,
               'max_on_power': 3998, 'mean': 700,
               'std': 1000, 's2s_length': 128,
               'houses': [1, 5], 'channels': [10, 18],
               'train_build': [1], 'valid_build': [5],
               'test_build': 2, 'test_channel': 8
               },
    'microwave': {'windowlength': 128,'on_power_threshold': 1000,
                  'max_on_power': 3969,'mean': 500,
                  'std': 800,'s2s_length': 128,
                  'houses': [1, 5],'channels': [13, 23],
                  'train_build': [1],'valid_build': [5],
                  'test_build': 2,'test_channel': 15},
    'fridge': { 'windowlength': 512,'on_power_threshold': 50,
                'max_on_power': 3323,'mean': 200,
                'std': 400,'s2s_length': 512,
                'houses': [1, 5],'channels': [12, 19],
                'train_build': [1],'valid_build': [5],
                'test_build': 2,'test_channel': 14},
    'dishwasher': {'windowlength': 512,'on_power_threshold': 10,
                   'max_on_power': 3964,'mean': 700,
                   'std': 1000,'s2s_length': 1536,
                   'houses': [1, 5],'channels': [6, 22],
                   'train_build': [1],'valid_build': [5],
                   'test_build': 2,'test_channel': 13},
    'washingmachine': {'windowlength': 599,'on_power_threshold': 20,
                       'max_on_power': 3999,'mean': 400,
                       'std': 700,'s2s_length': 2000,
                       'houses': [1, 5],'channels': [5, 24],
                       'train_build': [1],'valid_build': [5],
                       'test_build': 2,'test_channel': 12},
    'boiler': {'windowlength': 599,'on_power_threshold': 20,
               'max_on_power': 3999,'mean': 400,
               'std': 700,'s2s_length': 2000,
               'houses': [1],'channels': [2],
               'train_build': [1],'valid_build': [5],
               'test_build': 2,'test_channel': 12},
    'toaster': {'windowlength': 599,'on_power_threshold': 20,
                'max_on_power': 3999,'mean': 400,
                'std': 700,'s2s_length': 2000,
                'houses': [1],'channels': [11],
                'train_build': [1],'valid_build': [5],
                'test_build': 2,'test_channel': 12},
    'hair_dryer': {'windowlength': 599,'on_power_threshold': 20,
                   'max_on_power': 3999,'mean': 400,
                   'std': 700, 's2s_length': 2000,
                   'houses': [1],'channels': [39],
                   'train_build': [1],'valid_build': [5],
                   'test_build': 2,'test_channel': 12},
    'iron': {'windowlength': 599,'on_power_threshold': 20,
             'max_on_power': 3999,'mean': 400,
             'std': 700,'s2s_length': 2000,
             'houses': [1],'channels': [41],
             'train_build': [1], 'valid_build': [5],
             'test_build': 2, 'test_channel': 12},
    'gas oven': {'windowlength': 599,'on_power_threshold': 20,
                 'max_on_power': 3999,'mean': 400,
                 'std': 700,'s2s_length': 2000,
                 'houses': [1], 'channels': [42],
                 'train_build': [1],'valid_build': [5],
                 'test_build': 2,'test_channel': 12}
}
# todo: Each building_info comes from the label of each dataset, please modify it as needed to verify
"""For UKDale, we only use house1 house2 house5, so here is only the information for these three houses"""
UKDale_building_info = {
    'house_1': {'total': 'channel_1.dat','kettle': 'channel_10.dat',
               'microwave': 'channel_13.dat', 'fridge': 'channel_12.dat',
               'dishwasher': 'channel_6.dat', 'washingmachine': 'channel_5.dat'},

    'house_2': {'total': 'channel_1.dat','kettle': 'channel_8.dat',
               'microwave': 'channel_15.dat', 'fridge': 'channel_14.dat',
               'dishwasher': 'channel_13.dat', 'washingmachine': 'channel_12.dat'},

    'house_5': {'total': 'channel_1.dat','kettle': 'channel_18.dat',
               'microwave': 'channel_23.dat', 'fridge': 'channel_19.dat',
               'dishwasher': 'channel_22.dat', 'washingmachine': 'channel_24.dat'},
}

"""REDD"""
REDD_building_info = {
    # It should be noted that the total signal of each house on Redd here will have two tags (channel 1 and channel 2)
    # After our test and verification, these two labels should add up to the real total power signal of the house
    # However, the author did not do this for the sake of convenience, but found out the "sub-total signal"
    # corresponding to each electrical appliance(just consider fridge, microwave and dishwasher),
    # When performing NILM, you only need to read the electrical signal and the corresponding "sub-total signal" to
    # verify the algorithm, refer to the 'if-else-else' at the end

    # dishwasher：house1-main-2，app-6；house2-main-1，app-10；house3-main-2，app-9
    # fridge：house1-main-1，app-5；house2-main-2，app-9；house3-main-2，app-7
    # microwave：house1-main-1，app-11；house2-main-2，app-6；house3-main-1，app16
    'house_1': {'total': 'channel_2.dat', 'fridge': 'channel_5.dat',
                'dishwasher': 'channel_6.dat', 'microwave': 'channel_11.dat',  # channel_10 值很小
               },
    'house_2': {'total': 'channel_1.dat', 'fridge': 'channel_9.dat',
                'dishwasher': 'channel_10.dat', 'microwave': 'channel_6.dat',
               },
    'house_3': {'total': 'channel_2.dat', 'fridge': 'channel_7.dat',
                'dishwasher': 'channel_9.dat', 'microwave': 'channel_16.dat',
               },
    'house_4': {'total': 'channel_1.dat',  # no fridge
                'dishwasher': 'channel_15.dat',  # no microwave
               },
    'house_5': {'total': 'channel_2.dat', 'fridge': 'channel_18.dat',
                'dishwasher': 'channel_20.dat','microwave': 'channel_3.dat',
               },
    'house_6': {'total': 'channel_1.dat', 'fridge': 'channel_8.dat',
                'dishwasher': 'channel_9.dat',  # no microwave
               },
}
# dishwasher：house1-main-2，app-6；house2-main-1，app-10；house3-main-2，app-9
# fridge：house1-main-1，app-5；house2-main-2，app-9；house3-main-2，app-7
# microwave：house1-main-1，app-11；house2-main-2，app-6；house3-main-1，app16
if ap_name == 'dishwasher':
    REDD_building_info['house_1']['total'] = 'channel_2.dat'
    REDD_building_info['house_2']['total'] = 'channel_1.dat'
    REDD_building_info['house_3']['total'] = 'channel_2.dat'
elif ap_name == 'fridge':
    REDD_building_info['house_1']['total'] = 'channel_1.dat'
    REDD_building_info['house_2']['total'] = 'channel_2.dat'
    REDD_building_info['house_3']['total'] = 'channel_2.dat'
elif ap_name == 'microwave':
    REDD_building_info['house_1']['total'] = 'channel_1.dat'
    REDD_building_info['house_2']['total'] = 'channel_2.dat'
    REDD_building_info['house_3']['total'] = 'channel_1.dat'

"""REFIT counts the information of each house, please pay attention to the comment section,
 because some appliances are not present in some houses"""
REFIT_building_info = {
    'house_1': {'total': 'channel_0.dat',
                'fridge': 'channel_1.dat',  # todo:2.Chest-Freezer,3.Upright-Freezer
                'washingmachine':'channel_5.dat', 'dishwasher': 'channel_6.dat',
                # no microwave and kettle
                },
    'house_2': {'total': 'channel_0.dat',
                'fridge': 'channel_1.dat',  # todo:fridge-freezer
                'washingmachine': 'channel_2.dat','dishwasher': 'channel_3.dat',
                'microwave': 'channel_5.dat', 'kettle':'channel_8.dat'
                },
    'house_3': {'total': 'channel_0.dat',  # 经测试，此处channel 0的数据不正确，不包含fridge-freezer的信号
                'fridge': 'channel_2.dat',  # todo:2.fridge-freezer,3.freezer
                'washingmachine': 'channel_6.dat','dishwasher': 'channel_5.dat',
                'microwave': 'channel_8.dat', 'kettle':'channel_9.dat'
                },
    'house_4': {'total': 'channel_0.dat',
                'fridge': 'channel_1.dat',  # todo:1.fridge,2.freezer,3.fridge-freezer
                'washingmachine': 'channel_4.dat',
                # no dishwasher
                'microwave': 'channel_8.dat', 'kettle':'channel_9.dat'
                },
    'house_5': {'total': 'channel_0.dat',
                'fridge': 'channel_1.dat',  # todo:fridge-freezer
                'washingmachine': 'channel_3.dat', 'dishwasher': 'channel_4.dat',
                'microwave': 'channel_7.dat', 'kettle': 'channel_8.dat'
                },
    'house_6': {'total': 'channel_0.dat',
                'fridge': 'channel_1.dat',  # todo:freezer(Utility Room)
                'washingmachine': 'channel_2.dat','dishwasher': 'channel_3.dat',
                'microwave': 'channel_6.dat', 'kettle':'channel_7.dat'
                },
    'house_7': {'total': 'channel_0.dat',
                'fridge': 'channel_1.dat',  # todo:1.fridge,2.freezer(garage),3.freezer
                'washingmachine': 'channel_5.dat','dishwasher': 'channel_6.dat',
                # no microwave
                'kettle':'channel_9.dat'
                },
    'house_8': {'total': 'channel_0.dat',
                'fridge': 'channel_1.dat',  # todo:1.fridge,2.freezer
                'washingmachine': 'channel_4.dat',
                # no dishwasher
                'microwave': 'channel_8.dat', 'kettle':'channel_9.dat'
                },
    'house_9': {'total': 'channel_0.dat',
                'fridge': 'channel_1.dat',  # todo:fridge-freezer
                'washingmachine': 'channel_3.dat','dishwasher': 'channel_4.dat',
                'microwave': 'channel_6.dat', 'kettle':'channel_7.dat'
                },
    'house_10': {'total': 'channel_0.dat',
                 'fridge': 'channel_4.dat',  # todo:2.freezer,4.fridge-freezer
                 'washingmachine': 'channel_5.dat','dishwasher': 'channel_6.dat',
                 'microwave': 'channel_8.dat',
                 # no kettle
                 },
    'house_11': {'total': 'channel_0.dat',
                 'fridge': 'channel_1.dat',  # todo:1.fridge, 2.fridge-freezer
                 'washingmachine': 'channel_3.dat','dishwasher': 'channel_4.dat',
                 'microwave': 'channel_6.dat', 'kettle':'channel_7.dat'
                 },
    'house_12': {'total': 'channel_0.dat',
                 'fridge': 'channel_1.dat',  # todo:fridge-freezer
                 # no washingmachine and dishwasher
                 'microwave': 'channel_3.dat', 'kettle':'channel_4.dat'
                 },
    'house_13': {'total': 'channel_0.dat', # no fridge
                 'washingmachine': 'channel_3.dat','dishwasher': 'channel_4.dat',
                 'microwave': 'channel_8.dat', 'kettle':'channel_9.dat'
                 },
    'house_15': {'total': 'channel_0.dat',
                 'fridge': 'channel_1.dat',  # todo: fridge-freezer
                 'washingmachine': 'channel_3.dat','dishwasher': 'channel_4.dat',
                 'microwave': 'channel_7.dat', 'kettle':'channel_8.dat'
                 },
    'house_16': {'total': 'channel_0.dat',
                 'fridge': 'channel_1.dat',  # todo:1,2 are both fridge-freezer
                 'washingmachine': 'channel_5.dat','dishwasher': 'channel_6.dat',
                 # no microwave and kettle
                 },
    'house_17': {'total': 'channel_0.dat',
                 'fridge': 'channel_2.dat',  # todo:1.Freezer (Garage),2.Fridge-Freezer
                 'washingmachine': 'channel_4.dat',
                 # no dishwasher
                 'microwave': 'channel_7.dat', 'kettle':'channel_8.dat'
                 },
    'house_18': {'total': 'channel_0.dat',
                 'fridge': 'channel_1.dat',  # todo:1.Fridge(garage),2.Freezer(garage),3.Fridge-Freezer
                 'washingmachine': 'channel_5.dat','dishwasher': 'channel_6.dat',
                 'microwave': 'channel_9.dat',
                 # no kettle
                 },
    'house_19': {'total': 'channel_0.dat',
                 'fridge': 'channel_1.dat',  # todo: Fridge&Freezer
                 'washingmachine': 'channel_2.dat',
                 # no dishwasher
                 'microwave': 'channel_4.dat', 'kettle':'channel_5.dat'
                 },
    'house_20': {'total': 'channel_0.dat',
                 'fridge': 'channel_1.dat',  # todo:1.Fridge, 2.Freezer
                 'washingmachine': 'channel_4.dat','dishwasher': 'channel_5.dat',
                 'microwave': 'channel_8.dat', 'kettle':'channel_9.dat'
                 },
    'house_21': {'total': 'channel_0.dat',
                 'fridge': 'channel_1.dat',  # todo:Fridge-Freezer
                 'washingmachine': 'channel_3.dat','dishwasher': 'channel_4.dat',
                 # no microwave
                 'kettle':'channel_7.dat'  # todo:Kettle/Toaster
                 },
}