# RSTS
RSTS model

Code For RSTS model
1. 以前 500200 训练60K次 mean_rank =473.1886965376782

cross_similarity(->0) = 0.0323 

knn_precision(->1) = 59.685  效果不佳

2. 2021/11/15 重构该代码体系，准备用于AAAI22被接受后的代码上传

3. 该代码原为raw删除生成的data后的代码，先更改代码结构分为不同的文件夹

4. 2021/11/16晚上 检测Region.py中时空编码值转化是否存在问题, 从经纬度+时间->时空格子编码没有问题

5. 2021/11/20晚上，完成对region.py的正确性检测

6. 2021/11/21 开始更新new1，来自new, 并上传github
   更新注释和部分参数名，使其更符合规范

7. 2021/11/23 开始进行baseline的更新和评价指标的重构
   修正为在数据预处理时同时生成 训练数据+实验数据

8. 2021/12/09 新增实验测试数据最多为200K条， Region.py将针对一个region时空格子设定，同时生成训练集，验证集和实验测试用数据
20mins 生成100K个， porto大概需要6小时

9. 将实验评估中的距离函数全部转移至distance.py中，重新构造实验评估源文件 res_evaluate.py

200, 400, porto (2021/12/10)
current time:Thu Dec  9 22:55:11 2021
Iteration: 3000
Train Generative Loss: 0.902
Train Discriminative Cross Loss: 0.318
Train Discriminative Inner Loss: 0.102
Train Loss: 0.661
best_train_loss: 0.593
best_train_dis_loss: 0.528
best_val_loss: 58.673

current time:Fri Dec 10 08:20:39 2021
Iteration: 32000
Train Generative Loss: 0.353
Train Discriminative Cross Loss: 0.159
Train Discriminative Inner Loss: 0.023
Train Loss: 0.268
best_train_loss: 0.277
best_train_dis_loss: 0.139
best_val_loss: 30.099
验证检验: 100%|██████████| 195956/195956 [2:25:52<00:00, 22.39it/s]
current validate loss:  27.203080932874247
Saving the model at iteration 32000 validation loss 27.203080932874247

10. 改进实验baseline算法 EDR和EDwP, 重新获取实验结果

2022/4/3 开始准备注释用于IJCAI 23收录做准备

2022/4/11 改为多GPU训练
m0 = nn.DataParallel(m0, dim=1)
m1 = nn.DataParallel(m1)
m0.encoder等子模块需要变为m0.model.encoder
多GPU训练存在组合问题，改为单GPU

最近更改：2022/5/9
两个数据集均已经在3090-2上训练完成

2022/7/11 部署到自己电脑服务器运行

beijing100200 
beijing.h5中共有187358条轨迹
限制长度20-100 热度词频率30后，热度词共有23742，Region.py用时1.5h
写出有效测试集trip数目为:  37472

porto100300
使用前100万条轨迹，训练集80万，测试20万
词汇表建立完成，词汇表总量（包含4个特殊编码）：349124,Region.py用时约12h

porto_randTime 100300
使用前100万条轨迹，训练集80万，测试20万
词汇表建立完成，词汇表总量（包含4个特殊编码）：302390,Region.py用时约12h

将噪声偏移改为0.0005， 300
噪声偏移不会产生较大影响对算法表现
