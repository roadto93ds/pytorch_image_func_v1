### for classification / pathlist&label_map
class ImageDataset(Dataset):
    """
    path ::: data_dir/*** folder ***/img_file.xxx
    """
    def __init__(self, pathlist, transform, label_map):
        super().__init__()
        self.pathlist = pathlist
        self.label_map = label_map
        self.transform = transform 
    
    def __len__(self):
        return len(self.pathlist) 

    def __getitem__(self, index):
        img_path = self.pathlist[index]
        label_name = img_path.split("/")[-2]
        label = self.label_map[label_name]
        label = torch.tensor(label)
        img = self.transform(Image.open(img_path).convert("RGB"))
        return img, label

### for classification / dataframe(column: img_path & label)
class ImageDataset(Dataset):
    def __init__(self, df, transform):
        super().__init__()
        self.df = df
        self.transform = transform 
    
    def __len__(self):
        return len(self.df) 

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = self.transform(Image.open(row.img_path).convert("RGB"))
        label = torch.tensor(row.label)
        return img, label
      
