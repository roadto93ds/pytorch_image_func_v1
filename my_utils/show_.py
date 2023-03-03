### basic
def tensor_to_img(image_t):
    """
    image_t:torch.Tensor
    display one image
    tensor -> display image
    """
    image = image_t.cpu().detach().numpy().transpose((1,2,0)) # CHW -> HWC
    image = np.clip(image, 0, 1) # 0~1clip
    plt.imshow(image)
    plt.show()
    
def tensor2image_np(image_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Normalize tensor -> inverse_transform -> np.array
    Args:
        image_t: A pytorch Tensor
    Return:
        image_np
    """
    image_np = image_t.cpu().detach().numpy().transpose((1,2,0))
    mean = np.array(mean)
    std = np.array(std)
    image_np = std * image_np + mean # inverse_transform
    image_np = np.clip(image_np, 0, 1)
    return image_np

### for img pathlist
def filelist_to_imgs(file_list, rows=5, columns=10, scale=4, order=True):
    """
    display rows*columns images
    file_list:path_list
    scale:display scaling value
    order:path_order True/False
    """
    plt.figure(figsize=(scale*columns, scale*rows))
    if order == True:
        for i, img_path in enumerate(file_list[0:rows*columns]):
            img = Image.open(img_path).convert('RGB')
            plt.subplot(rows, columns, i+1)
            plt.imshow(img)
        plt.show()

    if order == False:
        for i, img_path in enumerate(random.sample(file_list, rows*columns)):
            img = Image.open(img_path).convert('RGB')
            plt.subplot(rows, columns, i+1)
            plt.imshow(img)
        plt.show()

        
### for simple classification
def dataset_to_imgs(dataset, rows=3, columns=5, scale=3, order=True):
    """
    display rows*columns images
    file_list:path_list
    scale:display scaling value
    order:dataset_order True/False
    """
    # index を作る。0番目からdatasetの長さ。
    dataset_index = list(range(len(dataset)))

    plt.figure(figsize=(scale*columns, scale*rows))
    if order == True:   
        for i, dataset_index in enumerate(dataset_index[0: rows*columns]):
            img = dataset[dataset_index][0]
            img = img.cpu().detach().numpy().transpose(1,2,0)
            img = np.clip(img, 0, 1) # 0~1clip
            ax = plt.subplot(rows, columns, i+1)
            plt.imshow(img)
        plt.show()

    if order == False:
        for i, dataset_index in enumerate(random.sample(dataset_index, rows*columns)):
            img = dataset[dataset_index][0]
            img = img.cpu().detach().numpy().transpose(1,2,0)
            img = np.clip(img, 0, 1) # 0~1clip
            ax = plt.subplot(rows, columns, i+1)
            plt.imshow(img)
        plt.show()

### for simple classification
def dataloader_to_imgs(dataloader, net=None, device=None):
    """
    display dataloader_imags(max:50）
    net:classification_model
    title:true_label / predict
    """
    images, labels = next(iter(dataloader))
    n_size = min(len(images), 50) # n_size: len(images) or 50

    # has classification_model
    if net is not None:
        inputs, labels = images.to(device), labels.to(device)
        # predict
        outputs = net(inputs)
        predicted = torch.max(outputs, 1)[1]
        images, labels, predicted = images.to("cpu"), labels.to("cpu"), predicted.to("cpu")

    plt.figure(figsize=(20,15))
    for i in range(n_size):
        ax = plt.subplot(5,10, i+1)
        if net is not None:
            if labels[i].item() == predicted[i].item():
                c = "k"
            else:
                c="b"
            ax.set_title(str(labels[i].item()) + "/" + str(predicted[i].item()) , c=c, fontsize=20)
            # ax.set_title(str(labels[i].item()) + "/" , c=c, fontsize=20)
        else:
            ax.set_title(labels[i].item(), fontsize=20) # .item -> delete tensor
        image_np = images[i].numpy().copy()
        img = np.transpose(image_np, (1,2,0))
        img = np.clip(img, 0,1)
        ax.set_axis_off()
        plt.imshow(img)
    plt.show()
