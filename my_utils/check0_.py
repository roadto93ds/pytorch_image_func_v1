### pre check

def pathlist_detail(pathlist:list, label_folder=False):
    """
    label_folder = True
    path ::: data_dir/*** folder ***/img_file.xxx
    """
    ext_list = [os.path.splitext(path_)[1] for path_ in pathlist]
    print(f"pathlist_len = {len(pathlist)}")
    print(collections.Counter(ext_list))
    if label_folder:
        name_list = [path_.split("/")[-2] for path_ in pathlist]
        print(collections.Counter(name_list))

def df_detail(df):
    print(df.shape)
    print(df.columns)
