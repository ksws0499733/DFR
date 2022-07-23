class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset in ['iRailway','iRailway_online']:
            return r'/home/user1106/DataSet/all_dataset2'
        if dataset in ['test','test_online']:
            return r'./doc'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
