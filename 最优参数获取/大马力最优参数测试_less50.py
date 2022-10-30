#读取数据
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # 能快速读取常规大小的文件。Pandas能提供高性能、易用的数据结构和数据分析工具
from sklearn.utils import shuffle  # 随机打乱工具，将原有序列打乱，返回一个全新的顺序错乱的值
from collections import OrderedDict
import xlwt
import os

##input column
cols = 6

loss_list = []



data1 = pd.read_excel('data/数据集-大马力.xlsx',sheet_name = '大马力', usecols=[1,2,3,4,5,6,7],header = 0, encoding='utf-8')
data1_1 = data1[data1['当年销量'] <= 50]
data1_2 = data1[data1['当年销量'] > 50]




data2 = pd.read_excel('data/数据集-大马力.xlsx',sheet_name = '预测用', usecols=[0,1,2,3,4,5,6,7],header = 0, encoding='utf-8')
for i in range(len(data2)):
    if data2['实际值'][i] < 1:
        data2['实际值'][i] = 1
data2 = data2[data2['实际值'] >= 5]
data2_1 = data2[data2['当年销量'] <= 50]
data2_2 = data2[data2['当年销量'] > 50]
data2_3 = data2[data2['当年销量'] <= 50]
data2_4 = data2[data2['当年销量'] > 50]


data2_1.drop(['省份'], axis=1,inplace=True)
data2_2.drop(['省份'], axis=1,inplace=True)



data_test1=data2_1.values
data_test2=data2_2.values


def get_test_data(data_test):
    time_step=1
    data_test=data_test
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size-1):
        x=normalized_test_data[i*time_step:(i+1)*time_step,:cols]
        y=normalized_test_data[i*time_step:(i+1)*time_step,cols]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:cols]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,cols]).tolist())
    return mean,std,test_x,test_y


def normal(df):
    # x_data为归一化后的前12列特征数据
    x_data = df[:,:-1]
    y_data = df[:,-1]
    mean_y = np.mean(y_data,axis=0)
    std_y = np.std(y_data,axis=0)
    
    x_data=(x_data-np.mean(x_data,axis=0))/np.std(x_data,axis=0)  #标准化
    y_data=(y_data-np.mean(y_data,axis=0))/np.std(y_data,axis=0)  #标准化
    return x_data, y_data,mean_y, std_y






def model(learning_rate):
    # 模型定义
    # 定义特征数据和标签数据的占位符
    # shape中None表示行的数量未知，在实际训练时决定一次带入多少行样本，从一个样本的随机SDG到批量SDG都可以
    x = tf.placeholder(tf.float32, [None, cols], name="X")  # 12个特征数据（12列）
    y = tf.placeholder(tf.float32, [None, 1], name="Y")  # 1个标签数据（1列）

    # 定义模型函数
    # 定义了一个命名空间.
    # 命名空间name_scope，Tensoflow计算图模型中常有数以千计节点，在可视化过程中很难一下子全部展示出来/
    # 因此可用name_scope为变量划分范围，在可视化中，这表示在计算图中的一个层级
    with tf.name_scope("Model"):
        # w 初始化值为shape=(12,1)的随机数
        w = tf.Variable(tf.random_normal([cols, 1], stddev=0.01), name="W")

        # b 初始化值为1.0
        b = tf.Variable(1.0, name="b")
        # w和x是矩阵相乘，用matmul,不能用mutiply或者*
        def model(x, w, b):
            return tf.matmul(x, w) + b
        # 预测计算操作，前向计算节点
        pred = model(x, w, b)

    # 定义均方差损失函数
    # 定义损失函数
    with tf.name_scope("LossFunction"):
        loss_function = tf.reduce_mean(tf.pow(y - pred, 2))  # 均方误差

    # 创建优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
    return loss_function, optimizer, pred, x, y, b, w

 
def run_model(loss_function, optimizer, pred, x_data, y_data, x, y, b, w,train_epochs):

    # 设置日志存储目录
    logdir = './test-3'
    # 创建一个操作，用于记录损失值loss，后面在TensorBoard中SCALARS栏可见
    sum_loss_op = tf.summary.scalar("loss", loss_function)

    # 把所有需要记录摘要日志文件的合并，方便一次性写入
    merged = tf.summary.merge_all()

    loss_list = []  # 用于保存loss值的列表
    Add_train_y = []
    Add_predict_y = []
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    module_file = tf.train.latest_checkpoint('./model-3')
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir, sess.graph)
        sess.run(tf.global_variables_initializer())
#         saver.restore(sess, module_file)
        for epoch in range(train_epochs):
            loss_sum = 0.0
            for xs, ys in zip(x_data, y_data):
                xs = xs.reshape(1, cols)
                ys = ys.reshape(1, 1)
                # feed数据必须和Placeholder的shape一致
                _, summary_str, loss = sess.run([optimizer, sum_loss_op, loss_function], feed_dict={x: xs, y: ys})
                writer.add_summary(summary_str, epoch)
                loss_sum = loss_sum + loss
            #   loss_list.append(loss)     #每步添加一次
            # 打乱数据顺序，防止按原次序假性训练输出
            x_data, y_data = shuffle(x_data, y_data)

            b0temp = b.eval(session=sess)  # 训练中当前变量b值
            w0temp = w.eval(session=sess)  # 训练中当前权重w值
            loss_average = loss_sum / len(y_data)  # 当前训练中的平均损失

            loss_list.append(loss_average)  # 每轮添加一次
#             print("epoch=", epoch + 1, "loss=", loss_average, "b=", b0temp, "w=", w0temp)

            Add_train_y.append(np.sqrt(mean_squared_error(y_data, sess.run(pred, feed_dict={x: x_data}))))
            if epoch % 200==0:
                print("保存模型：",saver.save(sess,'./model-3/stock2.model',global_step=epoch))


        


def prediction(pred,x,time_step,data_test): 
    pre = []
    true = []
    error = []
    error_abs = []
    mean,std,test_x,test_y=get_test_data(data_test)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('./model-3')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
            prob=sess.run(pred,feed_dict={x:test_x[step]})
            predict=prob.reshape((-1))
            test_predict.extend(predict) 
        test_y=np.array(test_y)*std[cols]+mean[cols]
        test_predict=np.array(test_predict)*std[cols]+mean[cols]
        
        accp1=np.average(np.abs(test_predict-test_y[:len(test_predict)]))
        accp=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差

        acc=np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)]#偏差
        acc_noabs = (test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)]
        for i in range(len(test_predict)):
            pre.append(round(test_predict[i],2))
            true.append(test_y[i])
            error.append(round(acc_noabs[i],1))
            error_abs.append(round(acc[i],1))

        
        return true,pre,error, error_abs


def plot_acc(df):
    t = []
    o_abs = []
    o = []

    o1 = df['实际值']
    o2 = df['pre']
    o_abs.append(abs(o1-o2)/o1)
    tmp = (o1-o2)/o1
    return tmp
        




def main1():
    lr = [0.005,0.01,0.015,0.02,0.025]
    echo = [400,700,1000,1300,1600]
    all = [data1_1,data1_2]
    data_test_ori = [data2_3,data2_4]
    data_test = [data_test1,data_test2]
    data_out = {}
    error_out = {}
    each = 0    
#     for i,j in enumerate(all):
    for m in lr:
        for n in echo:
            train_epochs = n
            learning_rate = m 
            repeat = 0
            error_out[each] = pd.DataFrame()
            range_err = pd.DataFrame()
            ##range_err['马力区间'] = ['25-40','40-50','50-60','60-70','70-80','80-90','90-100']
            range_err['马力区间'] = data_test_ori[0]['马力']
            for k in range(5):
                data = all[0].values
                x_data, y_data, mean_y, std_y = normal(data)
                loss_function, optimizer, pred, x, y, b, w = model(learning_rate)
                run_model(loss_function, optimizer, pred, x_data, y_data, x, y, b, w,train_epochs)
                true,pre,error, error_abs = prediction(pred,x,1,data_test[0])
                data_out[repeat] = data_test_ori[0]
                data_out[repeat]['pre'] = pre
                print (data_out[repeat])
#                 print(data_out[repeat])
                err = plot_acc(data_out[repeat])
                error_out[each] = range_err
                error_out[each]['%s-error_lr_%s_ech%s'%(repeat,m,n)] = err
                
                
                repeat += 1
#             print(error_out[each])    
            error_out[each].loc['mean'] = error_out[each].mean()
            mmn = error_out[each].loc['mean'][1:]
            mmn_list = " ".join([str(round(i,3)) for i in mmn])
            log.write("*"*40)
            log.write("\n%s %s 组合下五次平均误差是 %s\n"%(m,n,mmn_list))
            log.write("%s %s 组合下总平均误差是 %s\n"%(m,n,round(mmn.mean(),3)))
            log.write("*"*40)
            log.write("\n")
            log.flush()
            each += 1
            print (each,"次", m,n,"组合训练完成")
            

    
    writer = pd.ExcelWriter('大马力最优参数_less_50.xls',engine='xlsxwriter')
    start_row = 1
    for i in range(25):
        error_out[i].to_excel(writer, sheet_name='sheet1', startrow=start_row)
        start_row = start_row + error_out[i].shape[0]+2
    writer.save()
        


if __name__ == '__main__':
    if os.path.exists('./大马力_less_50.txt'):
        os.remove('大马力最优参数_less_50.txt')
    log = open('大马力最优参数_less_50.txt', 'a+')
    main1()
        





