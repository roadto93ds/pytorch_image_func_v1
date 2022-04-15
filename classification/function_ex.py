# make_pathlist
def make_pathlist(filepath, RGB_check = True):
  """
	sample
  rootpath = "/content/drive/MyDrive/0.PyTorch_GANs入門/data/horse2zebra/"
  filepath = os.path.join(rootpath + "trainA/*.jpg")
	make_pathlist(filepath)
  """
  # RGB次元かをチェックして、RGB次元の画像のみ格納（デフォルト）
  if RGB_check == True:
    path_list = [ path for path in glob.glob(filepath) if Image.open(path).mode == "RGB"]
    
  # RGBの確認しないで格納（モノクロも混ざる可能性あり）
  else:
    path_list = [ path for path in glob.glob(filepath)]

  return path_list
