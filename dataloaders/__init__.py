
import torch.utils.data as tdata
from torch.utils.data import DataLoader
from dataloaders.utils import collate_fn


def make_data_loader(args, **kwargs):

    print('\n------make_data_loader------\n')

    if args.dataset.endswith('online'):
        from .datasets.iRailway import build_online as iRaily_build
    else:
        from .datasets.iRailway import build_offline as iRaily_build
        
    num_class = 4
    print('  dataset: iRailway0')
    print('  num_class:',num_class)
    dataset_full = iRaily_build(state='train', args=args)

    dataset_test = iRaily_build(state='test', args=args)

    train_size = int(0.8 * len(dataset_full))
    
    val_size = len(dataset_full) - train_size

    dataset_train, dataset_val = tdata.random_split(dataset_full, [train_size, val_size])
    print('dataset_train',len(dataset_train))
    print('dataset_val',len(dataset_val))

    train_loader = DataLoader(dataset_train, 
                                batch_size=args.batch_size, 
                                shuffle=True,
                                collate_fn=collate_fn)
    val_loader = DataLoader(dataset_val, 
                                batch_size=args.batch_size, 
                                shuffle=False,
                                collate_fn=collate_fn)
    test_loader = DataLoader(dataset_test, 
                                batch_size=args.batch_size, 
                                shuffle=False,
                                collate_fn=collate_fn)


    return train_loader, val_loader, test_loader, num_class

