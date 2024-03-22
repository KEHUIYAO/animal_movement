import os
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
            mae_interpolation = f.read()

            # extract the mae value from 'Test MAE: 123.456'
            mae_interpolation = mae_interpolation.split(': ')[1]

        # with open(f'./results/{deer_id}/transformer/mae.txt') as f:
        #     mae = f.read()
        #     print(mae)

        with open(f'./results/{deer_id}/csdi/mae.txt') as f:
            mae_csdi = f.read()
            mae_csdi = mae_csdi.split(': ')[1]

        deer_list.append(deer_id)
        mae_interpolation_list.append(mae_interpolation)
        mae_csdi_list.append(mae_csdi)

    except:
        pass


# make a dataframe using deer_list, mae_interpolation_list, mae_csdi_list
df = pd.DataFrame({'deer_id': deer_list, 'mae_interpolation': mae_interpolation_list, 'mae_csdi': mae_csdi_list})

# count how many times mae_csdi is smaller than mae_interpolation
count = 0
for i in range(len(df)):
    if float(df['mae_csdi'][i]) < float(df['mae_interpolation'][i]):
        count += 1

# print the count
print('CSDI is better than interpolation:', count, 'out of', len(df), 'times')


# save the dataframe to a csv file
df.to_csv('results.csv', index=False)