class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'my':
            return r'/home/user1106/DataSet/Qingzang/Data_for_SCNN'
        elif dataset == 'mycls3':
            return r'/home/user1106/DataSet/Railway_class3'
        elif dataset == 'mycls5':
            return r'/home/user1106/DataSet/neardata5'
        elif dataset == 'mycls5_4d':
            return r'/home/user1106/DataSet/neardata5_4d'
        elif dataset == 'djicls5':
            return r'/home/user1106/DataSet/Railway_class5'
        elif dataset == 'mnist':
            return r'D:\论文相关\pytorch-deeplab-xception-master\data_set'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
