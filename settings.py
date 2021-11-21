
import argparse, os, torch, warnings, h5py
warnings.filterwarnings("ignore")
os.environ['CUDA_ENABLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = False

PAD = 0
BOS = 1
EOS = 2
UNK = 3
city_code = 2
cityname = "beijing" if city_code == 1 else "porto"
scale = 0.001
time_size = 400


def set_args():

    parser = argparse.ArgumentParser(description="train.py")
    '''
    *************************************************************************
    训练参数
    *************************************************************************
    '''
    parser.add_argument("-max_num_line", type=int, default=8000, help='读取训练集大小')
    parser.add_argument("-read_val_nums", type=int, default=10000, help="读取的验证集大小")
    parser.add_argument("-iter_num", default=3000000, help="总的训练迭代次数")
    parser.add_argument("-max_length", default=200, help="The maximum length of the target sequence")
    '''
    *************************************************************************
     Region参数
    *************************************************************************
    '''
    parser.add_argument("-cityname", default=cityname, help="city name")
    parser.add_argument("-scale", type=float, default=scale, help="city scale")
    parser.add_argument("-time_size", type=float, default=time_size, help="time span nums")
    parser.add_argument("-data", default=os.path.join('../data', cityname, cityname + str(int(scale * 100000))
                                                      + str(time_size)), help="训练集和模型存储目录")
    parser.add_argument("-checkpoint", default=os.path.join('../data', cityname,cityname + str(int(scale * 100000)) + str(time_size), 'checkpoint.pt'), help="checkpoint存放目录")
    parser.add_argument("-hot_freq", type=int, default=20, help="hot cell frequency, 计数出现这么多次认为是热度词")
    '''
    *************************************************************************
    神经网络层参数
    *************************************************************************
    '''
    parser.add_argument("-isToy", default= True, help="Set True if we are using toy data")
    parser.add_argument("-timeWeight", default= 0.2, help="The weight used to evaluate the temporal similarity")
    parser.add_argument("-need_rank_val", default= False, help="是否需要rank验证")
    parser.add_argument("-use_discriminative", action="store_true", default=True, help="Use the discriminative loss if the argument is given")
    parser.add_argument("-discriminative_w", type=float, default=0.3, help="discriminative loss weight")
    parser.add_argument("-dis_freq", type=float, default=10, help="discriminative loss weight 使用频率")
    parser.add_argument("-prefix", default="exp", help="Prefix of trjfile")
    parser.add_argument("-pretrained_embedding", default=None, help="Path to the pretrained word (cell) embedding")
    parser.add_argument("-num_layers", type=int, default=3, help="Number of layers in the RNN cell")
    parser.add_argument("-bidirectional", type=bool, default=True, help="True if use bidirectional rnn in encoder")
    parser.add_argument("-hidden_size", type=int, default=256, help="The hidden state size in the RNN cell")
    parser.add_argument("-embedding_size", type=int, default=256, help="The word (cell) embedding size")
    parser.add_argument("-dropout", type=float, default=0.2, help="The dropout probability")
    parser.add_argument("-max_grad_norm", type=float, default=5.0, help="The maximum gradient norm")
    parser.add_argument("-learning_rate", type=float, default=0.001)
    parser.add_argument("-batch", type=int, default=20, help="The batch size") 
    parser.add_argument("-generator_batch", type=int, default=32, help="The maximum number of words to generate each time.The higher value, the more memory requires.")
    parser.add_argument("-t2vec_batch", type=int, default=256, help="""The maximum number of trajs we encode each time in t2vec""")
    parser.add_argument("-start_iteration", type=int, default=0)
    parser.add_argument("-epochs", type=int, default=15, help="The number of training epochs")
    parser.add_argument("-print_freq", type=int, default=20000, help="Print frequency")
    parser.add_argument("-save_freq", type=int, default=20000, help="Save frequency")
    parser.add_argument("-cuda", type=bool, default=True, help="True if we use GPU to train the model")
    parser.add_argument("-criterion_name", default="KLDIV",help="NLL (Negative Log Likelihood) or KLDIV (KL Divergence)")
    parser.add_argument("-dist_decay_speed", type=float, default=0.8,
        help="""How fast the distance decays in dist2weight, a small value will give high weights for cells far away""")
    parser.add_argument("-bucketsize", default=[(30, 50), (50, 70), (80, 100)],help="Bucket size for training")
    args = parser.parse_args()
    return args


