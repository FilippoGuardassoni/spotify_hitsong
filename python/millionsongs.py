# Million Songs Dataset
# See Million Songs Dataset website for instructions

# usual imports
import os
import sys
import glob
import hdf5_getters as getters
import pandas as pd

# path to the Million Song Dataset subset ( uncompressed )
# CHANGE IT TO YOUR LOCAL CONFIGURATION
msd_subset_path = '/Users/amministratore/Desktop/MillionSongSubset'
msd_subset_data_path = os.path.join('data', msd_subset_path)
msd_subset_addf_path = os.path.join('additionalFiles', msd_subset_path)
assert os.path.isdir(msd_subset_path), 'wrong path'  # sanity check

# path to the Million Song Dataset code
msd_code_path = '/Users/amministratore/PycharmProjects/hitsong'
assert os.path.isdir(msd_code_path), 'wrong path'  # sanity check

# add some paths to python so we can import MSD code
sys.path.append(os.path.join(msd_code_path, 'PythonSrc'))

# define this very useful function to iterate the files
def apply_to_all_files(basedir, func=lambda x: x, ext='.h5'):
    cnt = 0
    # iterate over all files in all subdirectories
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*'+ext))
        # count files
        cnt += len(files)
        # apply function to all files
        for f in files:
            func(f)
    return cnt

# count the number of files in the dataset
print('number of song files:', apply_to_all_files(msd_subset_data_path))

year = []
title = []
artist_name = []
artist_id = []

# define the function to apply to all files (get the selected features)
def get_all(filename):
            h5 = getters.open_h5_file_read(filename)
            year.append(getters.get_year(h5))
            title.append(getters.get_title(h5))
            artist_name.append(getters.get_artist_name(h5))
            artist_id.append(getters.get_artist_id(h5))
            h5.close()

# apply the previous function to all files
apply_to_all_files(msd_subset_data_path, func=get_all)
print('Found:', len(year), 'records')  # check

# convert the lists in a dataframe and export to csv
df = pd.DataFrame(list(zip(year, title, artist_name, artist_id)), columns=['Year', 'Title', 'Artist Name', 'Artist ID'])
df.to_csv('millionsongs.csv', header=True)
print(df)
