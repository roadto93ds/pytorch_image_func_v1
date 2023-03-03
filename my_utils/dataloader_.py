from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler

### base1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

### base2:shuffle=True
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(dataset), num_workers=num_workers, pin_memory=True)
### base2:shuffle=False
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset), num_workers=num_workers, pin_memory=True)

### WeightedRandomSampler ::: for classification (imbalanced)
### sampler = get_weighted_sampler(sample_weights, replacement_=True)
### dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)

def get_sample_weights(dataset):
    """
    sample_weights : assign class weights by dataset
    len(sample_weights) = len(dataset)
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
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)
    """
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=replacement_)
    return sampler
