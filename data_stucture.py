import pandas as pd
import csv

# adding M/F
block1_mf = ['M', 'F', 'M', 'M', 'M', 'F', 'M', 'F', 'F', 'F', 'F', 'F', 'F', 'M', 'F', 'F', 'F', 'F', 'F', 'M', 'F',
             'F', 'M', 'F', 'F', 'F', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'F', 'F', 'F', 'M', 'M']
block1_io = ['I', 'O', 'I', 'I', 'I', 'O', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'O', 'I', 'O', 'O',
             'O', 'O', 'I', 'O', 'I', 'O', 'O', 'O', 'I', 'I', 'I', 'I', 'I', 'I', 'O', 'I', 'O', 'I', 'O', 'I']
block2_mf = ['M', 'F', 'F', 'M', 'M', 'F', 'F', 'F', 'F', 'F', 'M', 'F', 'F', 'M', 'F', 'M', 'M', 'F', 'F', 'F', 'F',
             'M', 'M', 'F', 'F', 'M', 'F', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F']
block2_io = ['O', 'O', 'I', 'O', 'I', 'I', 'O', 'O', 'O', 'I', 'I', 'O', 'I', 'I', 'O', 'I', 'O', 'O', 'I', 'I', 'I',
             'I', 'O', 'I', 'I', 'O', 'I', 'O', 'O', 'O', 'O', 'I', 'O', 'I', 'I', 'O', 'I', 'I', 'O', 'I', 'O']


# block3 = []
# block4 = []
# block5 = []
# block6 = []
# block7 = []
# block8 = []
#
# df1 = pd.DataFrame(columns=['ID', 'M/F', 'I/O', 'Trigger Response'])
# df1.index.name = 'Index'
# # print(df1)
#
# df1['M/F'] = block1_mf
# df1['I/O'] = block1_io
# # print(df1)
#
# df2 = pd.DataFrame(columns=['ID', 'M/F', 'I/O', 'Trigger Response'])
# df2.index.name = 'Index'
# df2['M/F'] = block2_mf
# df2['I/O'] = block2_io
# # print(df2)
#
# # to add dataframes to right of another:
#
# df_out = pd.concat([df1, df2], axis=1)  # cant have one of these dataframes to csv before this step
#
# print(df_out)


# df_out = pd.concat([my_array, image_df], axis=1)

# def create_df(block_num, target_id):
#     df = pd.DataFrame(columns=['Target', 'M/F', 'I/O', 'Trigger Response'])
#     df['Target'] = [target_id] * 41
#     df['M/F'] = block1_mf
#     df['I/O'] = block1_io
#     print(df)
#
#
# create_df(block_num=1, target_id='M')

in_list1 = [1,2,3,4]
in_list2 = [12,13,14,15]
out_list = [in_list1, in_list2]
print(out_list)

df = pd.DataFrame(columns=['List 1 Vals', 'List 2 Vals'])
df.loc[len(df.index)] = out_list

print(df)

df.to_csv('out.csv', encoding='utf-8', index_label='Index')
