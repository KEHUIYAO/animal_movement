import os
deer_id_list = os.listdir('./results/')

# remove .DS_Store
deer_id_list = [int(x) for x in deer_id_list if x != '.DS_Store']

for deer_id in sorted(deer_id_list):
    print(deer_id)

    # enter the interpolation folder, and read mae.txt file
    try:
        with open(f'./results/{deer_id}/interpolation/mae.txt') as f:
            mae = f.read()
            print(mae)

        with open(f'./results/{deer_id}/transformer/mae.txt') as f:
            mae = f.read()
            print(mae)
    except:
        pass


