### basic
def get_pathlist(data_dir):
    """
    path ::: data_dir/*** folder ***/img_file.xxx
    """
    filepath = os.path.join(data_dir, "**/*")
    pathlist = glob.glob(filepath)
    return pathlist

### for classification 
def get_label_dict(data_dir):
    """
    path ::: data_dir/*** folder ***/img_file.xxx
    """
    class_name = os.listdir(data_dir)
    label_ = list(range(len(class_name)))
    return dict(zip(class_name, label_))

def get_df(data_dir):
    """
    path ::: data_dir/*** folder ***/img_file.xxx
    return df:pandas
        column : path & label
    """
    img_pathlist = get_pathlist(data_dir)
    label_map = get_label_dict(data_dir)
    label_list = [label_map[path.split("/")[-2]] for path in img_pathlist]
    df = pd.DataFrame({
                "path" : img_pathlist,
                "label" : label_list
                })
    return df
