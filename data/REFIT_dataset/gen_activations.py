import matplotlib.pyplot as plt
import _pickle
from DataSet import ActivationGenerator
from PreDefine import UKDale_building_info,REDD_building_info,REFIT_building_info,ap_name,data_name,bVAE_dict
house_dict = bVAE_dict
ratio = 10/100  # The first 10% of the time period data was used to extract the activation signal
if data_name == 'REDD':
    act_name = 'redd_act.pickle'  # The name of the generated activation signal pickle file
    building_info = REDD_building_info
    appliance_names = ['fridge', 'dishwasher', 'microwave']
    Tr_name = 'house_2'
    Te_name = 'house_1'
    Va_name = 'house_3'
elif data_name == 'UKDale':
    act_name = 'ukdale_act.pickle'  # The name of the generated activation signal pickle file
    building_info = UKDale_building_info
    appliance_names = ['washingmachine', 'dishwasher', 'microwave', 'fridge', 'kettle']
    Tr_name = 'house_1'
    Te_name = 'house_2'
    Va_name = 'house_5'
elif data_name == 'REFIT':
    act_name = 'refit_act.pickle'  # The name of the generated activation signal pickle file
    building_info = REFIT_building_info
    appliance_names = ['fridge','washingmachine', 'dishwasher', 'microwave',  'kettle']
    Tr_name = house_dict[ap_name][0]
    Te_name = house_dict[ap_name][1]
    Va_name = house_dict[ap_name][2]


def generate_act(Temp):
    Temp.appliance_data = Temp.read_app_data(
        file_location=Temp.file_location,
        appliance_file_name=Temp.appliance_file_name,
        separator=Temp.separator,
        types=Temp.types
    )
    (Temp.appliance_data,
     Temp.app_data_name) = Temp.resample_app_data(
        index_name=Temp.col_names[0],
        appliance_data=Temp.appliance_data)
    Temp.appliance_data = Temp.remove_abnormal_data(Temp.appliance_data)
    # Refrigerator signals that exceed normal values are removed
    if Temp.app_name == 'fridge':
        Temp.appliance_data = Temp.remove_big_abnormal()
    # todo: When debugging, if you want to display the signal, you can use the following code in the terminal
    # plt.figure()
    # plt.plot(Temp.appliance_data)
    # plt.title('after remove abnormal')
    # plt.show()
    return Temp


if __name__ == '__main__':

    dic_tr = {}
    dic_te = {}
    dic_va = {}
    Dict = {}

    house1_filename = data_name+'/'+Tr_name
    house2_filename = data_name+'/'+Te_name
    house3_filename = data_name+'/'+Va_name
    i = 0
    for name in appliance_names:
        i = i+1
        print(i, 'th appliance', name, ' activation signal')
        tr_house = ActivationGenerator(
            file_location=house1_filename,
            mixed_file_name=building_info[Tr_name]['total'],
            appliance_file_name=building_info[Tr_name][name],
            app_name=name
        )
        tr_house.dataloader_name = Tr_name
        if name == ap_name:
            te_house = ActivationGenerator(
                file_location=house2_filename,
                mixed_file_name=building_info[Te_name]['total'],
                appliance_file_name=building_info[Te_name][name],
                app_name=name
            )
            te_house.dataloader_name = Te_name
        va_house = ActivationGenerator(
            file_location=house3_filename,
            mixed_file_name=building_info[Va_name]['total'],
            appliance_file_name=building_info[Va_name][name],
            app_name=name
        )
        va_house.dataloader_name = Va_name

        tr_house = generate_act(tr_house)
        if name == ap_name:
            te_house = generate_act(te_house)
        va_house = generate_act(va_house)
        print('tr start……')
        tr_house.appliance_data = tr_house.get_activations(chunk=tr_house.appliance_data, percent=ratio)
        print('te start……')
        if name == ap_name:
            te_house.appliance_data = te_house.get_activations(chunk=te_house.appliance_data, percent=ratio)
        print('va start……')
        va_house.appliance_data = va_house.get_activations(chunk=va_house.appliance_data, percent=ratio)

        dic_tr[name] = tr_house.activation_data
        if name == ap_name:
            dic_te[name] = te_house.activation_data
        dic_va[name] = va_house.activation_data
        print(name, 'is finished')

    Dict['tr'] = dic_tr
    Dict['te'] = dic_te
    Dict['va'] = dic_va

    _pickle.dump(Dict, open(act_name, 'wb'))
    print('Congratulations！Generate activations finished')
