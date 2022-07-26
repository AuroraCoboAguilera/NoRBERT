'''

Data Loader for output obtained from Transformers

Author: Aurora Cobo Aguilera
Date: 1 April 2020

'''

from HDF5Dataset import HDF5Dataset

class TransformerOutputDataset(HDF5Dataset):
    def __init__(self, dataroot, option=1):
        '''
        :param dataroot: root of the dataset to load
        :param option: Add functions to apply  some transformations to the data if it is neccesary
        '''

        if option == 1:
            super().__init__(dataroot, transform=None)


    def transform_1(self, x):
        x = x.reshape(self.dim0*self.dim1)
        return x

