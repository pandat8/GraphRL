from utils.utils import open_dataset


train, val, test = open_dataset('./data/ERGcollection/erg_small.pkl')

print(train.__len__())
