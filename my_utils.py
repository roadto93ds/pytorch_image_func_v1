def check_ext(data_dir):
    """
    data_dirの直下のフォルダのみ（.txt などのファイルは除外される）を参照して
    フォルダ内の拡張子のunique数を返す
    """
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        # print(os.path.isdir(folder_path))
        if os.path.isdir(folder_path):
            ls_list = os.listdir(folder_path)
            ext_list = [ os.path.splitext(ls_)[1] for ls_ in ls_list ]
            count_result = collections.Counter(ext_list)
            print(f"***** {folder_name} *****")
            print(count_result)
        else:
            print(f"***** {folder_name} *****")
            print("is_not_folder")

def get_sample_weights(dataset):
    """
    sample_weights : assign class weights by dataset
        dataset の labelの数 から 確定する
    """
    label_list = [data_[1].item() for data_ in dataset]
    n_class = np.unique(np.array(label_list))
    count_list = [label_list.count(class_) for class_ in n_class]

    class_weights = list()
    for count_ in count_list:
        if count_ > 0:
            class_weights.append(1/count_)

    sample_weights = [0] * len(dataset)
    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    return sample_weights

def get_weighted_sampler(sample_weights, replacement_ = True):
    """
    return : WeightedRandom DataLoader
        oversampling : replacement_ = True
        not oversampling : replacement_ = False
    """
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=replacement_)
    return sampler
