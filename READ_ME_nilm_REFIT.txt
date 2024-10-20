1. there is a difference between linux and windws for data processing in 'DataSet'.  

DataLoader.remove_abnormal_data():
……
        mix_data = np.reshape(mix_data, (-1, 1))  # todo: for windows local
        # mix_data = np.reshape(mix_data.to_numpy(), (-1, 1))  # for linux server
        app_data = np.reshape(app_data, (-1, 1))  # todo: for windows local
        # app_data = np.reshape(app_data.to_numpy(), (-1, 1))  # for linux server
……

2. you need to pay attention to the 'todo' ANNOTATION in the code, which means you can 
do some operations to understand the code.

3. you download REFIT data from this site, then run our demo.py file to preprocess it.

4. After finishing the preparation work for the Python environment and moving the preprocessed REFIT house data folder to the REFIT folder , you should run 'gen_activations', 'gen_dataset_by_activations'
'save_data_by_batch', 'train_appname', 'test_appname'.
if you cannot train, you can just run 'gen_activations', 'gen_dataset_by_activations', 'test_appname'





