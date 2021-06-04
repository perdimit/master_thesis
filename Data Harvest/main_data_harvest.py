from NN_EW_modules import NN_EW_data_handling as dh
import pexpect
import sys
import os

seed = 42

# targets_configs = [["1000023_1000024_13000_NLO_1", "1000023_-1000024_13000_NLO_1", "1000022_1000022_13000_NLO_1", "1000023_1000037_13000_NLO_1", "1000035_1000024_13000_NLO_1", "1000025_1000024_13000_NLO_1", "1000025_-1000024_13000_NLO_1"
#                     , "1000022_-1000037_13000_NLO_1"]]
targets_configs = [["1000023_1000024_13000_NLO_1", "1000023_-1000024_13000_NLO_1", "1000022_1000022_13000_NLO_1"]]
target = targets_configs[0]
filename, data = dh.get_data(targets_configs=targets_configs,
                       directory='/home/per-dimitri/Dropbox/Master/test_data/EWonly/', rewrite=False, return_df=True, get_params='PMCSXUV')

masses, save_string, load_stored = dh.names_creation(target)
data = dh.preprocess(data, target, masses)[0]
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

print(train_data.head(5).to_string())

filename_train, filename_test = filename + '_train', filename + '_test'
train_data.to_csv(filename_train, sep='\t')
# train_data.to_txt(filename_train, sep=" ")
# train_data.to_txt(filename_test, sep=" ")
test_data.to_csv(filename_test, sep='\t')
data.to_csv(filename + '_filt', sep='\t')

def move_file_to_model():
    current_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    train_path = current_path + filename_train
    test_path = current_path + filename_test
    print(test_path)
    path_model = os.path.dirname(os.getcwd()) + '/Models/'
    if os.path.exists(path_model):
        os.replace(train_path, path_model + filename_train)
        os.replace(test_path, path_model + filename_test)
# move_file_to_model()

###### HPC Upload ######
def upload_to_hpc(adress, filename, filename_add=None):
    if filename_add is None:
        filename_out = filename
    else:
        filename_out = filename + filename_add
    dir_path = os.path.dirname(os.path.realpath(__file__))
    pw = input("Enter you password: ")
    child = pexpect.spawn("scp %s %s:~/%s" % (filename, adress, filename_out), cwd=dir_path, timeout=30)
    child.logfile_read = sys.stdout.buffer
    child.expect("%s's password:" % adress)
    child.sendline(pw)
    child.expect(pexpect.EOF)
    child.close()


# upload_to_hpc(adress='perdimis@saga.sigma2.no', filename=filename_train, filename_add=None)
