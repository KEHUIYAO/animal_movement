import os
import re
import pandas as pd
deer_id_list = os.listdir('./results/')

# remove .DS_Store
deer_id_list = [int(x) for x in deer_id_list if x != '.DS_Store']
deer_list = []
mae_interpolation_list = []
mae_transformer_list = []
mae_csdi_list = []

for deer_id in sorted(deer_id_list):
    # print(deer_id)

    # enter the interpolation folder, and read mae.txt file
    try:
        with open(f'./results/{deer_id}/interpolation/mae.txt') as f:
            text = f.read()

            # extract the mae value from 'Test MAE: 123.456\nTest MRE' using regex
            specific_number = re.search(r'Test MAE: (\d+\.\d+|\d+)', text)
            mae_interpolation = float(specific_number.group(1)) if specific_number else None


        # with open(f'./results/{deer_id}/transformer/mae.txt') as f:
        #     mae = f.read()
        #     print(mae)

        with open(f'./results/{deer_id}/csdi/mae.txt') as f:
            text = f.read()
            # extract the mae value from 'Test MAE: 123.456\nTest MRE' using regex
            specific_number = re.search(r'Test MAE: (\d+\.\d+|\d+)', text)
            mae_csdi = float(specific_number.group(1)) if specific_number else None

        deer_list.append(deer_id)
        mae_interpolation_list.append(mae_interpolation)
        mae_csdi_list.append(mae_csdi)

    except:
        pass


# make a dataframe using deer_list, mae_interpolation_list, mae_csdi_list
df = pd.DataFrame({'deer_id': deer_list, 'mae_interpolation': mae_interpolation_list, 'mae_csdi': mae_csdi_list})

# count how many times mae_csdi is smaller than mae_interpolation, and average decrease in mae

count = 0
decrease = 0
for i in range(len(df)):
    if float(df['mae_csdi'][i]) < float(df['mae_interpolation'][i]):
        count += 1
        decrease += float(df['mae_interpolation'][i]) - float(df['mae_csdi'][i])

# print the count
print('CSDI is better than interpolation:', count, 'out of', len(df), 'times')
print('Average decrease in MAE:', decrease / count)

# what is the mean of the difference between mae_interpolation and mae_csdi
df['mae_diff'] = df['mae_interpolation'] - df['mae_csdi']
print('Mean of the difference between mae_interpolation and mae_csdi:', df['mae_diff'].mean())


# save the dataframe to a csv file
df.to_csv('results.csv', index=False)