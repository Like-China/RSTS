import random
import torch
import os
from trainer.models import EncoderDecoder
from loader.data_utils import pad_arrays_keep_invp







## 生成子轨迹
def subtrj_generate(trj_origin, r1, r2, vocal_size):
    import copy
    '''
    :param trj: 原轨迹Tb (list)
    :param r1: 噪音率
    :param r2: 缺失率
    :      vocal_size:词汇表大小
    :return: 子轨迹
    '''
    trj = copy.deepcopy(trj_origin)
    for i in range(len(trj)):
        add_num = int(len(trj[i]) * r1)
        erase_num = int(len(trj[i]) * r2)
        for k in range(erase_num):
            erase_positon = random.randrange(0, len(trj[i]), 1)
            trj[i].pop(erase_positon)
        for k in range(add_num):
            insert_position = random.randrange(0, len(trj[i]), 1)
            # 0-4031
            insert_num = random.randrange(0, vocal_size, 1)
            trj[i].insert(insert_position, insert_num)
    return trj

    

# 输入：轨迹集合T
# 输出：对应的向量表示
# 注意T中的轨迹数需要大于 args.t2vec_batch
def t2vec(args,T):
    "read source sequences from trj.t and write the tensor into file trj.h5"
    m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        m0.load_state_dict(checkpoint["m0"])
        if torch.cuda.is_available():
            m0.cuda()
        m0.eval()
        
        vecs = []
        
        for ii in range(len(T)//args.t2vec_batch):
            # src 该组最大轨迹长度*num_seqs(该组轨迹个数) 
            src, lengths, invp = pad_arrays_keep_invp(T[ii*args.t2vec_batch:(ii+1)*args.t2vec_batch])
            if src is None: break
            if torch.cuda.is_available():
                src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
            h, _ = m0.encoder(src, lengths) # 【层数*双向2，该组轨迹个数，隐藏层数】【6，10，128】
            ## (num_layers, batch, hidden_size * num_directions) 【3，10，256】
            h = m0.encoder_hn2decoder_h0(h)
            ## (batch, num_layers, hidden_size * num_directions) 【10，3，256】
            h = h.transpose(0, 1).contiguous()
            ## (batch, *)
            #h = h.view(h.size(0), -1)
            vecs.append(h[invp].cpu().data)
        ## (num_seqs, num_layers, hidden_size * num_directions)
        
        vecs = torch.cat(vecs) # [10,3,256]
        # ## 存储三层 输出的隐藏层结构，每一层是 batch个256维的向量
        ## (num_layers, num_seqs, hidden_size * num_directions)
        vecs = vecs.transpose(0, 1).contiguous()  ## [3,10,256]
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
    return vecs[m0.num_layers-1]
    


