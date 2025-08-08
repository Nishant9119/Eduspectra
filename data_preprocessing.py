import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
#KDD CUP 15 dataset files
course_df = pd.read_csv("kddcup15\\date.csv")
enrollment_df = pd.read_csv("C:\\Users\\Adros\\Desktop\\nishant\\early_dropouts\\kddcup15\\test\\test\\enrollment_test.csv")
logs_df = pd.read_csv("C:\\Users\\Adros\\Desktop\\nishant\\early_dropouts\\kddcup15\\test\\test\\log_test.csv")
truth_df = pd.read_csv("C:\\Users\\Adros\\Desktop\\nishant\\early_dropouts\\kddcup15\\test\\test\\truth_test.csv")
#logs tru or inner_merged is merged file of truth_df and logs_df on enrollment_id and how = 'inner'
#date_enrollment_combined_csv is merged df of course_df and enrollment_df on course_id and how ='inner'.

logs_tru = pd.read_csv("test_inner_merged.csv")
df1=pd.read_csv("test_date_enrollment_combined.csv")
combined_data = pd.read_csv("test_combined_data_tbu.csv")
df = pd.read_csv("test_data_combined_time_corrected.csv")

logs_tru['enrollment_id'] = logs_tru['enrollment_id'].astype(str)
df1['enrollment_id'] = df1['enrollment_id'].astype(str)
logs_tru['enrollment_id'] = logs_tru['enrollment_id'].str.strip()
df1['enrollment_id'] = df1['enrollment_id'].str.strip()
final_combined_data = pd.merge(logs_tru,df1,how='inner',on='enrollment_id')
final_combined_data.to_csv("test_final_combined_data.csv")
fdf = pd.read_csv("C:\\Users\\Adros\\OneDrive\\Desktop\\Early Dropout Predction in E-learning Courses\\test_final_combined_data.csv")
new_data = fdf[['enrollment_id', 'time', 'source',
       'event', 'truth', 'course_id', 'from', 'to']]
new_data.to_csv("test_combined_data_tbu.csv",index =False)
combined_data = pd.read_csv("C:\\Users\\Adros\\OneDrive\\Desktop\\Early Dropout Predction in E-learning Courses\\test_combined_data_tbu.csv")

#get data of only one candidate
for i in df['enrollment_id'].unique():
    one_candidate_data =  df[df['enrollment_id']==i]
    one_candidate_data.to_csv("one_candidate_data.csv",index = False)
    break

#To check how many unique users are there because some users may have enrolles to more than one course
df1 = pd.read_csv("C:\\Users\\Adros\\OneDrive\\Desktop\\Early Dropout Predction in E-learning Courses\\test_date_enrollment_combined.csv")
print(len(df1['username'].unique()))
print(len(df1['enrollment_id'].unique()))

#write the time column in week's value according to when the course started
#Trying it with one student's data first and then to complete data
one_student_data = pd.read_csv("C:\\Users\\Adros\\OneDrive\\Desktop\\Early Dropout Predction in E-learning Courses\\one_candidate_data.csv")
one_student_data['time'] = pd.to_datetime(one_student_data['time'],format = '%Y-%m-%dT%H:%M:%S')
one_student_data['from'] = pd.to_datetime(one_student_data['from'],format = "%d-%m-%Y")
one_student_data['time'] = one_student_data['time'].dt.strftime('%d-%m-%Y')
one_student_data['from'] = one_student_data['from'].dt.strftime('%d-%m-%Y')
one_student_data['time'] = pd.to_datetime(one_student_data['time'],format = '%d-%m-%Y')
one_student_data['from'] = pd.to_datetime(one_student_data['from'],format = '%d-%m-%Y')
one_student_data['days_since_start'] = (one_student_data['time']-one_student_data['from']).dt.days
one_student_data['week_label'] =  np.ceil((one_student_data['days_since_start']+1)/7).astype(int)
one_student_data.to_csv("ks.csv",index = None)

df = pd.read_csv("C:\\Users\\Adros\\OneDrive\\Desktop\\Early Dropout Predction in E-learning Courses\\ks.csv")
source_counts = df.groupby(['week_label','source']).size().unstack(fill_value = 0).reset_index()
source_counts.rename(columns=  {'browser':'browser_visits','server':'server_visits'},inplace=True)
event_counts = df.groupby(['week_label','event']).size().unstack(fill_value=0).reset_index()
final_table = source_counts.merge(event_counts,on='week_label',how= 'left')
final_table['enrollment_id'] = df['enrollment_id'][0]
final_table['truth'] = df['truth'][0]
print(final_table)

student_learning_matrix = final_table.drop(columns = [['week_label','browser_visits','server_visits','enrollment_id','truth']])
print(student_learning_matrix)

combined_data['time'] = pd.to_datetime(combined_data['time'],format = '%Y-%m-%dT%H:%M:%S')
combined_data['from'] = pd.to_datetime(combined_data['from'],format = "%d-%m-%Y")
combined_data['time'] = combined_data['time'].dt.strftime('%d-%m-%Y')
combined_data['from'] = combined_data['from'].dt.strftime('%d-%m-%Y')
combined_data['time'] = pd.to_datetime(combined_data['time'],format = '%d-%m-%Y')
combined_data['from'] = pd.to_datetime(combined_data['from'],format = '%d-%m-%Y')
combined_data.to_csv("data_combined_time_corrected.csv",index = None)

final_stud_learning_matrix = []
df = pd.read_csv("C:\\Users\\Adros\\OneDrive\\Desktop\\Early Dropout Predction in E-learning Courses\\data_combined_time_corrected.csv")
required_columns = [
    'week_label', 'browser_visits', 'server_visits',
    'access_count', 'navigation_count', 'wiki_count', 
    'problem_count', 'page_close_count', 'video_count', 
    'discussion_count', 'enrollment_id', 'truth'
]
for i in df['enrollment_id'].unique():
    student_data = df[df['enrollment_id'] == i].copy()
    student_data['time'] = pd.to_datetime(student_data['time'], format='%Y-%m-%d')
    student_data['from'] = pd.to_datetime(student_data['from'], format='%Y-%m-%d')
    student_data['days_since_start'] = (student_data['time'] - student_data['from']).dt.days
    student_data['week_label'] = np.ceil((student_data['days_since_start'] + 1) / 7).astype(int)
    weeks_template = pd.DataFrame({'week_label': range(1, 6)})
    source_counts = student_data.groupby(['week_label', 'source']).size().unstack(fill_value=0).reset_index()
    source_counts.rename(columns={'browser': 'browser_visits', 'server': 'server_visits'}, inplace=True)
    source_counts = weeks_template.merge(source_counts, on='week_label', how='left').fillna(0)
    event_counts = student_data.groupby(['week_label', 'event']).size().unstack(fill_value=0).reset_index()
    event_counts.rename(columns={
            "access": 'access_count', 'navigate': 'navigation_count',
            'wiki': 'wiki_count', 'problem': 'problem_count',
            'page_close': 'page_close_count', 'video': 'video_count',
            'discussion': 'discussion_count'
        }, inplace=True)
    event_counts = weeks_template.merge(event_counts, on='week_label', how='left').fillna(0)
    final_table = source_counts.merge(event_counts, on='week_label', how='left').fillna(0)
    final_table['enrollment_id'] = i
    final_table['truth'] = student_data['truth'].iloc[0]
    for col in required_columns:
        if col not in final_table.columns:
            final_table[col] = 0  
    final_stud_learning_matrix.append(final_table)
    print(f"appended enrollment id {i}")
final_df = pd.concat(final_stud_learning_matrix,ignore_index=True)
final_df.to_csv("Final_full_data_for_rank.csv",index =False)

#Finding M Ratio Matrix

week_list = [1,2,3,4,5]
max_access = []
max_nav = []
max_wiki = []
max_prob = []
max_page = []
max_video = []
max_disc = []
max_dict=  {}
df_m = pd.read_csv("test_Final_full_data_for_rank.csv")
for i in df_m['week_label'].unique():
    slist = []
    dfw = df_m[df_m['week_label']==i]
    print(f"Maximum Access count in week {i} is : {max(dfw['access_count'])}")
    print(f"Maximum Navigation count in week {i} is : {max(dfw['navigation_count'])}")
    print(f"Maximum Wiki count in week {i} is : {max(dfw['wiki_count'])}")
    print(f"Maximum Problem count in week {i} is : {max(dfw['problem_count'])}")
    print(f"Maximum Page_Close count in week {i} is : {max(dfw['page_close_count'])}")
    print(f"Maximum Video count in week {i} is : {max(dfw['video_count'])}")
    print(f"Maximum Discussion count in week {i} is : {max(dfw['discussion_count'])}")
    max_access.append(max(dfw['access_count']))
    max_nav.append(max(dfw['navigation_count']))
    max_wiki.append(max(dfw['wiki_count']))
    max_prob.append(max(dfw['problem_count']))
    max_page.append(max(dfw['page_close_count']))
    max_video.append(max(dfw['video_count']))
    max_disc.append(max(dfw['discussion_count']))
max_dict['week_label'] = week_list
max_dict['max_access_count'] = max_access
max_dict['max_navigation_count'] = max_nav
max_dict['max_wiki_count'] = max_wiki
max_dict['max_problem_count']=  max_prob
max_dict['max_page_close_count'] = max_page
max_dict['max_video_count'] = max_video
max_dict['max_discussion_count']=  max_disc
qq = pd.DataFrame(max_dict)
qq.to_csv("test_max_values.csv",index = False)

#Finding SRatio

week_list = [1,2,3,4,5]
avg_access = []
avg_nav = []
avg_wiki = []
avg_prob = []
avg_page = []
avg_video = []
avg_disc = []
avg_dict = {}
df_m = pd.read_csv("test_Final_full_data_for_rank.csv")
for i in df_m['week_label'].unique():
    dfw = df_m[df_m['week_label']==i]
    print(f"Average Access count in week {i} is : {dfw['access_count'].mean()}")
    print(f"Average Navigation count in week {i} is : {dfw['navigation_count'].mean()}")
    print(f"Average Wiki count in week {i} is : {dfw['wiki_count'].mean()}")
    print(f"Average Problem count in week {i} is : {dfw['problem_count'].mean()}")
    print(f"Average Page_Close count in week {i} is : {dfw['page_close_count'].mean()}")
    print(f"Average Video count in week {i} is : {dfw['video_count'].mean()}")
    print(f"Average Discussion count in week {i} is : {dfw['discussion_count'].mean()}")

    avg_access.append(dfw['access_count'].mean())
    avg_nav.append(dfw['navigation_count'].mean())
    avg_wiki.append(dfw['wiki_count'].mean())
    avg_prob.append(dfw['problem_count'].mean())
    avg_page.append(dfw['page_close_count'].mean())
    avg_video.append(dfw['video_count'].mean())
    avg_disc.append(dfw['discussion_count'].mean())

avg_dict['week_label'] = week_list
avg_dict['avg_access_count'] = avg_access
avg_dict['avg_navigation_count'] = avg_nav
avg_dict['avg_wiki_count'] = avg_wiki
avg_dict['avg_problem_count'] = avg_prob
avg_dict['avg_page_close_count'] = avg_page
avg_dict['avg_video_count'] = avg_video
avg_dict['avg_discussion_count'] = avg_disc
aa = pd.DataFrame(avg_dict)
aa.to_csv("test_average_values.csv",index = False)

#This Section of code will give final_rank_matrix.npy

import pandas as pd
import numpy as np

# The rank function
def rank(n, arr):
    if n == 0:
        return 0
    for i in range(len(arr)):
        if arr[i] == n:
            return (1 / (i + 1))  
df_m = pd.read_csv("test_Final_full_data_for_rank.csv")
week_list = [1, 2, 3, 4, 5]
final_matrix = []
for enroll_id in sorted(df_m['enrollment_id'].unique()):
    print(f"Processing Enrollment ID: {enroll_id}")
    student_week_ranks = []
    for week in week_list:
        dfw = df_m[df_m['week_label'] == week]
        dfg = dfw[dfw['enrollment_id'] == enroll_id]
        access_week = sorted(dfw['access_count'])
        nav_week = sorted(dfw['navigation_count'])
        wiki_week = sorted(dfw['wiki_count'])
        problem_week = sorted(dfw['problem_count'])
        page_week = sorted(dfw['page_close_count'])
        video_week = sorted(dfw['video_count'])
        discussion_week = sorted(dfw['discussion_count'])
        
        # Calculate ranks for this enrollment_id in the current week
        access_rank = rank(dfg['access_count'].values[0], access_week)
        nav_rank = rank(dfg['navigation_count'].values[0], nav_week)
        wiki_rank = rank(dfg['wiki_count'].values[0], wiki_week)
        problem_rank = rank(dfg['problem_count'].values[0], problem_week)
        page_rank = rank(dfg['page_close_count'].values[0], page_week)
        video_rank = rank(dfg['video_count'].values[0], video_week)
        discussion_rank = rank(dfg['discussion_count'].values[0], discussion_week)
        
        # Store the ranks for the current week
        week_ranks = [access_rank, nav_rank, wiki_rank, problem_rank, page_rank, video_rank, discussion_rank]
        student_week_ranks.append(week_ranks)
    
    # Append the 5x7 matrix for this enrollment_id to the final 3D matrix
    print(student_week_ranks)
    final_matrix.append(student_week_ranks)
    break
final_matrix = np.array(final_matrix)
print(f"Shape of final matrix: {final_matrix.shape}")
np.save("test_final_rank_matrix.npy", final_matrix)

#finding M Ratio Complete Matrix

max_list =[]
max_df  = pd.read_csv("test_max_values.csv")
nmax= max_df[[ 'max_access_count', 'max_navigation_count',
       'max_wiki_count', 'max_problem_count', 'max_page_close_count',
       'max_video_count', 'max_discussion_count']]
max_np = np.array(nmax).astype(float)
real_df = pd.read_csv("test_Final_full_data_for_rank.csv")
for i in sorted(real_df['enrollment_id'].unique()):
    print(f"Processing Enrollment  id : {i}")
    dfe = real_df[real_df['enrollment_id']==i][['access_count', 'navigation_count',
       'wiki_count', 'problem_count', 'page_close_count',
       'video_count', 'discussion_count']]
    np_dfe = np.array(dfe)
    max_array = np_dfe/max_np
    max_list.append(max_array)
final_np_array = np.array(max_list)
print(final_np_array.shape)
np.save("test_final_max_matrix.npy",final_np_array)


avg_list = []
avg_df = pd.read_csv("test_average_values.csv")
navg= avg_df[[ 'avg_access_count', 'avg_navigation_count',
       'avg_wiki_count', 'avg_problem_count', 'avg_page_close_count',
       'avg_video_count', 'avg_discussion_count']]
avg_np = np.array(navg).astype(float)
real_df = pd.read_csv("test_Final_full_data_for_rank.csv")
for i in sorted(real_df['enrollment_id'].unique()):
    print(f"Processing Enrollment  id : {i}")
    dfe = real_df[real_df['enrollment_id']==i][['access_count', 'navigation_count',
       'wiki_count', 'problem_count', 'page_close_count',
       'video_count', 'discussion_count']]
    avgnp_dfe = np.array(dfe)
    avg_array = np.where(avg_np!=0,avgnp_dfe/avg_np,0)
    avg_list.append(avg_array)
    print(avg_array)
final_np_array = np.array(avg_list)
print(final_np_array.shape)
np.save("test_final_avg_matrix.npy",final_np_array)

def normalize_with_clip(array):
    return np.clip(array * 255, 0, 255).astype(np.uint8)
max_mat = np.load("test_final_max_matrix.npy")
avg_mat=  np.load("test_final_avg_matrix.npy")
rank_mat= np.load("test_final_rank_matrix.npy")
count = 0
output_dir_d = "test_images/Dropouts"
output_dir_nd = "test_images/NonDropouts"
os.makedirs(output_dir_d, exist_ok=True)
os.makedirs(output_dir_nd, exist_ok=True)
for i in sorted(truth_df['enrollment_id'].unique()):
    print(f"processing enrollment id : {i}")
    dfn = truth_df[truth_df['enrollment_id']==i]
    print(dfn)
    if dfn['truth'].iloc[0]==1:
        print("truth is 1")
        array1 = max_mat[count]
        array2 = avg_mat[count]
        array3 = rank_mat[count]

        ar1_normalized = normalize_with_clip(array1)
        ar2_normalized = normalize_with_clip(array2)
        ar3_normalized = normalize_with_clip(array3)
        image = np.stack([ar1_normalized, ar2_normalized, ar3_normalized], axis=-1)
        save_path = os.path.join(output_dir_d, f"{i}.png")
        plt.imsave(save_path, image)
    elif dfn['truth'].iloc[0]==0:
        print(f"truth is 0 ")
        array1 = max_mat[count]
        array2 = avg_mat[count]
        array3 = rank_mat[count]

        ar1_normalized = normalize_with_clip(array1)
        ar2_normalized = normalize_with_clip(array2)
        ar3_normalized = normalize_with_clip(array3)
        image = np.stack([ar1_normalized, ar2_normalized, ar3_normalized], axis=-1)
        save_path = os.path.join(output_dir_nd, f"{i}.png")
        plt.imsave(save_path, image)
    count+=1
