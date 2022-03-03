from cgi import test
import splitfolders

def split_data(train_ratio=0.8, test_ratio=0.2)
    splitfolders.ratio("./data/05. 표준데이터(PRPD)", output="data/split_data/05. 표준데이터(PRPD)2", seed=1337, ratio=(train_ratio, .0, test_ratio))
