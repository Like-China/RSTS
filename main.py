from trainer.train import train
from settings import set_args
import time

if __name__ == "__main__":
    # porto读取需要半小时，beijing读取需要4分钟
    print("train："+str(time.ctime()))
    train(set_args())