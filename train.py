# for training with crossvalidation

import torch
import numpy as np
from dataset import getNycData
from transformers import AutoImageProcessor, Mask2FormerConfig, Mask2FormerForUniversalSegmentation
from torchgeo.samplers import RandomGeoSampler, RandomBatchGeoSampler
from torch.utils.data import DataLoader, TensorDataset, random_split, SubsetRandomSampler
from torchgeo.datasets import concat_samples, random_grid_cell_assignment
from sklearn.model_selection import KFold

def main():
    # torch CUDA support
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # hyperparameters
    BATCH_SIZE = 1
    test_split = 0.2
    K = 5 # cross_validation splits
    patch_size = 256

    SEED = 1701
    random_generator = torch.Generator()
    random_generator.manual_seed(SEED)

    processor = AutoImageProcessor.from_pretrained('facebook/mask2former-swin-large-cityscapes-semantic')
    config = Mask2FormerConfig.from_pretrained('facebook/mask2former-swin-large-cityscapes-semantic')
    model = Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-large-cityscapes-semantic').to(device)


    dataset = getNycData()
    # holdout split of 20% of the data
    train_val_datasets, test_dataset = random_grid_cell_assignment(
        dataset=dataset, 
        fractions=[1-test_split, test_split], 
        grid_size=10,
        generator=random_generator
    )

    kf_split_dataset = random_grid_cell_assignment(
        dataset=train_val_datasets, 
        fractions=np.full(K, 1/K), # cross validation
        grid_size=10,
        generator=random_generator
    )

    # train
    for k in range(K):
        
        
        
        print("FOLD", k+1)
        val_dataset = kf_split_dataset[k]
        train_datasets_arr = kf_split_dataset[:k] + kf_split_dataset[k+1:]
        print(train_datasets_arr)
        # merge all other folds into training dataset
        train_dataset = train_datasets_arr[0]
        for train_dataset_split in range(1, K-1):
            train_dataset = train_dataset | train_datasets_arr[train_dataset_split]
        
        # create dataloaders
        train_sampler = RandomBatchGeoSampler(
            dataset=train_dataset,
            size=patch_size,
            batch_size=BATCH_SIZE
            # length=1000 # from docs: defaults to approximately the maximal number of non-overlapping chips of size size that could be sampled from the dataset
        )    
        val_sampler = RandomBatchGeoSampler(
            dataset=val_dataset,
            size=patch_size,
            batch_size=BATCH_SIZE,
            # length=1000 # from docs: defaults to approximately the maximal number of non-overlapping chips of size size that could be sampled from the dataset
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)
        
        for train_batch in train_loader:
            print(train_batch)
            break
            

        
        print(train_loader, val_loader)


# for batch in dataloader:
#     batch = {image: mask.to(device) for image, mask in batch.items()}
#     print(batch)
#     break
#     outputs = model(**batch, mask_labels=batch['mask'])
#     print(outputs)
#     break

def train_step():
    pass

@torch.no_grad()
def val_step():
    pass

@torch.no_grad()
def test_step():
    pass

if __name__ == '__main__':
    main()