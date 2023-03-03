class ImageTransform():
    def __init__(self, resize):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize, resize))
            ])
    def __call__(self, img):
        return self.data_transform(img)

