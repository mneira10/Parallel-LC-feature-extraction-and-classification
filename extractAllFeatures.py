import extract
import tqdm
import pandas as pd 

def renameCols(cols):
    translate = {'flux':'Flux','magnitude':'Mag','flux_error':'Fluxerr','magnitude_error':'Magerr','mjd':'MJD','bandpass':'band'}
    for i,c in enumerate(cols):
        if c in translate: cols[i]=translate[c]
    return cols

def unique_ids_list(df_lcs):
    return df_lcs.index.get_level_values('ID').unique().format()

def print_num_ids_shape(df_lcs):
    unique_ids = unique_ids_list(df_lcs)
    print('Num IDs: {}  Shape: {}'.format(len(unique_ids), df_lcs.shape))

def filter_light_curves(df_lcs, min_obs):
    df_count = df_lcs.groupby('ID', as_index=True).count()
    df_count['ObsCount'] = df_count['Flux']
    df_count = df_count[['ObsCount']]
    df_lcs_with_counts = df_lcs.join(df_count, how='inner')
    # Remove objects with less than min_obs
    df_filtered = df_lcs_with_counts[df_lcs_with_counts.ObsCount >= min_obs]
    return df_filtered

def oversample(df_lcs, copies=0):
    df_oversample = df_lcs.copy()
    df_oversample['copy_num'] = 0
    for i in range(1, copies+1):
        df_temp = df_lcs.copy()
        df_temp['copy_num'] = i
        df_temp['Mag'] = np.random.normal(df_lcs.Mag, df_lcs.Magerr)
        df_oversample = df_oversample.append(df_temp)
    df_oversample = df_oversample.set_index(['copy_num'], append=True)
    return df_oversample


def extract_features(df_lcs):
#     pid = (current_process().name.split('-')[1])
#     print("Process ", pid, " starting...")
#     print("Process ", pid, " extracting num_copy...")
    # Extract num_copy list
    num_copy_list = df_lcs.index.get_level_values('copy_num').unique()
    num_copies = len(num_copy_list)
#     print("Process ", pid, " extracting id_list...")
    # Extract IDs list
    unique_ids_list = df_lcs.index.get_level_values('ID').unique()
    num_ids = len(unique_ids_list)
#     print("Process ", pid, " creating ouput vars...")
    # Create empty feature dict
    feats_dict = extract.feature_dict(30)
    feats_dict['ObsCount'] = []
    feats_dict['Class'] = []
    # Add 'ID' and 'copy_num' index lists
    index_id_list = []
    index_copy_num_list = []
#     print("Process ", pid, " starting processing loop...")
    num_objects = num_ids*num_copies
    for kasd,num_copy in enumerate(num_copy_list):
        for i, obj_id in enumerate(unique_ids_list):
            # Print status
            current_object_i = (num_copy+1)*(i+1)
            if(current_object_i%1000 == 0):
                print(current_object_i, '/', num_objects)
            # Get current object light curve
#             print(pid, current_object_i, 'geting object light curve')
            df_object = df_lcs.loc[obj_id, :, num_copy]
            
            # Get features
            try:
                obj_feats = extract.features(df_object, feats_dict)
            except:
                continue
            # Append features
            for k, v in obj_feats.items():
                feats_dict[k].append(obj_feats[k])

            # Append Indexes
            index_id_list.append(obj_id)
            index_copy_num_list.append(num_copy)

            # Append class and obs_count
            feats_dict['Class'].append(df_object['SN'].unique()[0])
            feats_dict['ObsCount'].append(df_object.shape[0])
            
            #TEMPORARYYYY---------------------------------
            # if i==4:
            #   break

    # Create feature dataframe
    print(len(index_id_list))
    print(len(index_copy_num_list))
    df_feats = pd.DataFrame(feats_dict).set_index([index_id_list, index_copy_num_list])
    df_feats.index.names = ['ID', 'copy_num']
    return df_feats

def save_features(df_feats, obj_type):
    outdir = FEATURES_PATH
    filename_raw = '{}.csv'
    filename = filename_raw.format(obj_type)
    df_feats.to_csv(outdir + filename)
