"""This demo can convert the original CSV data format of each house into various files in the house folder
Channel.dat file for use with subsequent gen activations, gen dataset by activations, etc
Perform data preprocessing and download CSV files from
https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned
Processed_Data_CSV.7z
Some areas may be handled roughly, but as they are already sufficient, there has been no further optimization"""
import csv
import os
import struct


def dealer(i):
    csv_name = 'House_'+i+'.csv'  # todo: origin 1-21.csv file
    folder_name = 'house_'+i  # todo: target 1-21
    num_signal = 10  # 10 signals to read
    # read CSV
    with open(csv_name, newline='') as csvfile:
        csvreader = csv.reader(csvfile)

        # first line
        first_row = next(csvreader)

        print(first_row)
        columns = zip(*csvreader)
        b = [0]*(num_signal+2)  # +2 first two signal is no use
        i = 0
        for column in columns:
            b[i] = column
            i = i+1
            # print(column)

        os.makedirs(folder_name, exist_ok=True)
        j = 2
        for k in range(len(b)-2):  # The first two columns are timestamps, so the number of loops is reduced by two
            channel_name = 'channel_'+str(j-2)+'.dat'
            dat_file = os.path.join(folder_name, channel_name)

            data_reordered = [b[1]] + [b[j]]
            j = j+1
            # Save the data as a. dat file by column, with a carriage return and space after each column of data
            with open(dat_file, 'wb') as file:
                for item in zip(*data_reordered):
                    for i, value in enumerate(item):
                        value_bytes = value.encode('utf-8')
                        file.write(struct.pack('15s', value_bytes))
                        if i < len(item) - 1:
                            file.write(b' ')  #
                    file.write(b'\n')  #


if __name__ == '__main__':
    lis = ['1','2','3','4','5','6','7','8','9','10','11','12','13','15','16','17','18','19','20','21']
    for i in lis:
        dealer(i)