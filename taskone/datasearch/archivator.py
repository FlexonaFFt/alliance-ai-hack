import gzip 
import shutil

with gzip.open('ctr_train.csv.gz', 'rb') as f_in:
    with open('ctr_train.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

with gzip.open('ctr_test.csv.gz', 'rb') as f_in:
    with open('ctr_test.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

with gzip.open('ctr_sample_submission.csv.gz', 'rb') as f_in:
    with open('ctr_sample_submission.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)