import os

def add_num_to_txt_name(text, format='.csv'):
    if len(format) >= 1:
        text = text[:-len(format)]
    text_ = text
    i = 0
    while os.path.exists(text_ + format):
        text_ = text
        text_ += '_(' + str(i) + ')'
        i += 1
        if i >= 20:
            print("More than 20 files with same name. Time to clean up!")
    return text_ + format

def best_model_params_from_pd():
    pass

def best_model_params_from_csv():
    pass