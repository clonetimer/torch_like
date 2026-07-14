"""Data loader factory — creates DataLoader for training/val/test splits."""

from torch.utils.data import DataLoader

from torch_like.data.dataset import Dataset_ETT_hour


def data_provider(args, flag):
    """Create DataLoader for the specified dataset and split.

    Args:
        args: namespace with data_path, data, batch_size, etc.
        flag: 'train', 'val', or 'test'.

    Returns:
        DataLoader, Dataset
    """
    if flag not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported data split: {flag!r}")
    shuffle_flag = flag == "train"
    batch_size = args.batch_size

    data_set = Dataset_ETT_hour(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        scale=True,
        freq=args.freq,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=flag == "train" and len(data_set) >= batch_size,
    )

    return data_set, data_loader
