from data_provider.data_holder import AnomalyDataset, AnomalyDataAugmenter
from torch.utils.data import DataLoader
import torch


def anomaly_collate_fn(batch):
    data = torch.stack([item[0] for item in batch])
    first_index = torch.tensor([item[1] for item in batch])
    return data, first_index


class AnomalyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, config, **kwargs):
        super().__init__(dataset, batch_size=batch_size, collate_fn=anomaly_collate_fn, **kwargs)
        self.augmenter = AnomalyDataAugmenter(args=config,
                                              numerical_column=dataset.numerical_column,
                                              categorical_column=dataset.categorical_column,
                                              replacing_data=dataset.replacing_data,
                                              full_data=dataset.data
                                              )

    def __iter__(self):
        for batch in super().__iter__():
            augmented_data, anomaly_labels = self.augmenter.augment_batch(batch)
            yield augmented_data, anomaly_labels


def data_provider(args, flag):
    dataset = AnomalyDataset(
        args,
        flag=flag
    )

    if args.augment and flag == 'train':
        data_loader = AnomalyDataLoader(
            dataset,
            batch_size=args.batch_size,
            config=args,
            shuffle=(flag == 'train'),
            drop_last=False
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False
        )

    return dataset, data_loader
