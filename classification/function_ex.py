def make_pathlist(filepath, RGB_check = True):
    """
        sample
    rootpath = "/content/drive/MyDrive/0.0_pytorch_baseline/pokemon/train"
    filepath = os.path.join(train_rootpath , "**/*.jpg")
    ** ：class_n folder (n=1~K)
        make_pathlist(filepath)
    RGB_check：eliminate_not3d image（default）
    """
    # check RGB
    if RGB_check:
        path_list = [ path for path in glob.glob(filepath) if Image.open(path).mode == "RGB"]
    # check RGB = False
    else:
        path_list = [ path for path in glob.glob(filepath)]
    return path_list


def make_pathlist(rootpath, get_ext):
    """
    rootpath = "/root/workspace/0.for_zemi/data/PokemonData"
    get_ext = [".jpeg", ".jpg", ".png"]
    """
    path_list = list()
    for ext in get_ext:
        filepath = os.path.join(rootpath, f"**/*{ext}")
        # path_listにする際に、 .mode == "RGB" で3次元構造になっていないものを弾く
        path_list += [path for path in glob.glob(filepath) if Image.open(path).mode == "RGB"]

    return  path_list
