import tarfile

with tarfile.open('train.tar.gz', 'r:gz') as tar:
    tar.extractall(path='train_folder')

with tarfile.open('test.tar.gz', 'r:gz') as tar:
    tar.extractall(path='test_folder')