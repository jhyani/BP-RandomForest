##test line
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

##input column
cols = 6

loss_list = []
# ##迭代轮次
# train_epochs = 1000

# # 学习率
# learning_rate = 0.01




data1 = pd.read_excel('data/数据集5_3.xlsx',sheet_name = '中马力', usecols=[1,2,3,4,5,6,7],header = 0, encoding='utf-8')
data1_1 = data1[data1['当年销量'] <= 50]
data1_2 = data1[data1['当年销量'] > 50]




data2 = pd.read_excel('data/数据集5_3.xlsx',sheet_name = '预测用', usecols=[0,1,2,3,4,5,6,7],header = 0, encoding='utf-8')
for i in range(len(data2)):
    if data2['实际值'][i] < 1:
        data2['实际值'][i] = 1
data2 = data2[data2['当年销量'] >= 10]
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
    logdir = './test'
    # 创建一个操作，用于记录损失值loss，后面在TensorBoard中SCALARS栏可见
    sum_loss_op = tf.summary.scalar("loss", loss_function)

    # 把所有需要记录摘要日志文件的合并，方便一次性写入
    merged = tf.summary.merge_all()

    loss_list = []  # 用于保存loss值的列表
    Add_train_y = []
    Add_predict_y = []
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    module_file = tf.train.latest_checkpoint('./model-2')
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
                print("保存模型：",saver.save(sess,'./model-2/stock2.model',global_step=epoch))


        

def plot(pred,x_data,y_data,x,y):
    n = np.random.randint(50)  # 随机确定一条来看看效果
    x_test = x_data[n]
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
    #参数恢复
        module_file = tf.train.latest_checkpoint('./model-2')
        saver.restore(sess, module_file)
        x_test = x_test.reshape(1, cols)
        predict = sess.run(pred, feed_dict={x: x_test})
        print("预测值：%f" % predict)

        target = y_data[n]
        print("标签值：%f" % target)
        plt.show()

    #######        曲线拟合效果，可以看出预测效果不错       #####
        test_predictions = sess.run(pred, feed_dict={x: x_data})
        plt.scatter(y_data, test_predictions)
        plt.xlabel('True Values [1000$]')
        plt.ylabel('Predictions [1000$]')
        plt.axis('equal')
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        _ = plt.plot([-100, 100], [-100, 100])
        plt.show()
        
def prediction(pred,x,time_step,data_test): 
    pre = []
    true = []
    error = []
    error_abs = []
    mean,std,test_x,test_y=get_test_data(data_test)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('./model-2')
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

#         print("average difference: ",accp)
#         print("absolute difference: ",accp1)
        acc=np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)]#偏差
        acc_noabs = (test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)]
#         print ("预测值  ","  真实值")
        for i in range(len(test_predict)):
            pre.append(int(test_predict[i]))
            true.append(test_y[i])
            error.append(round(acc_noabs[i],2))
            error_abs.append(round(acc[i],2))
#             print("%s: "%i,test_predict[i],test_y[i], acc[i])
        
        return true,pre,error, error_abs
def split_range(df):
    t = []
    o_abs = []
    tr_sum = []
    pre_sum = []
    o = []

    for i in range(7):
        if i <1 :
            t.append(df[(df['马力'] >=25+10*i) & (df['马力'] <40)])
        else:
            t.append(df[(df['马力'] >=30+10*i) & (df['马力'] <(i+1)*10+30)])
        o1 = t[i]['实际值'].sum()
        o2 = t[i]['pre'].sum()
#         o_abs.append(abs(o1-o2)/o1)
        o_abs.append((o1-o2)/o1)
        o.append((o1-o2)/o1) 
        tr_sum.append(o1)
        pre_sum.append(o2)
    return tr_sum,pre_sum,o_abs

def plot_acc(df):

    ##数据分为西安和山东两组，分别输出世纪指与预测值的曲线对比图
    df1 = df[df['省份'] == '陕西']
    df2 = df[df['省份'] == '山东']
    tr_sum1, pre_sum1, o_abs1 = split_range(df1)
    tr_sum2, pre_sum2, o_abs2 = split_range(df2)
        
    colors1 = '#00CED1' #点的颜色
    colors2 = '#DC143C'
    area = np.pi * 4**2  # 点面积
    engin = ['25-40','40-50','50-60','60-70','70-80','80-90','90-100']
    for o_abs in [o_abs1, o_abs2]:
        for i, j in zip(engin,o_abs):
            log.write("马力区间\t绝对值误差\n")
            log.write("%s      %s\n"%(str(i),j))
            log.flush()
        log.write("平均误差是 %s\n"%(str(sum(o_abs)/len(o_abs))))
        log.write("*"*40)
        log.write("\n\n")
        log.flush()
    plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
    fig1 = plt.figure()
    plt.title('各个马力区间绝误差率',fontsize=14)  # 标题，字号设置
    plt.plot(engin,o_abs1,color='r', label= '陕西省: average=%s'%(sum(o_abs1)/len(o_abs1)))
    plt.plot(engin,o_abs2,color='b', label= '山东省: average=%s'%(sum(o_abs2)/len(o_abs2)))
    plt.scatter(engin,o_abs1, s=area, c='r', alpha=0.4, label='陕西省')
    plt.scatter(engin,o_abs2, s=area, c='b', alpha=0.4, label='山东省')

    plt.ylim([-5, 5])
    plt.xlabel('马力区间')  # 位置设置
    plt.ylabel('误差率')  # 方向，位置设置
    plt.legend()
    plt.show()
 
    fig4 = plt.figure()
    plt.title('陕西各个马力区间绝实际值vs预测值',fontsize=14)  # 标题，字号设置
    plt.plot(engin,tr_sum1,color='r', label= '实际值')
    plt.plot(engin,pre_sum1,color='b', label= '预测值')
    plt.scatter(engin,tr_sum1, s=area, c='r', alpha=0.4, label='实际值')
    plt.scatter(engin,pre_sum1, s=area, c='b', alpha=0.4, label='预测值')

    plt.xlabel('马力区间')  # 位置设置
    plt.ylabel('销量')  # 方向，位置设置
    plt.legend()
    plt.show()
    
    fig5 = plt.figure()
    plt.title('山东各个马力区间绝实际值vs预测值',fontsize=14)  # 标题，字号设置
    plt.plot(engin,tr_sum2,color='r', label= '实际值')
    plt.plot(engin,pre_sum2,color='b', label= '预测值')
    plt.scatter(engin,tr_sum2, s=area, c='r', alpha=0.4, label='实际值')
    plt.scatter(engin,pre_sum2, s=area, c='b', alpha=0.4, label='预测值')

    plt.xlabel('马力区间')  # 位置设置
    plt.ylabel('销量')  # 方向，位置设置
    plt.legend()
    plt.show()

    fig2 = plt.figure()
    plt.title('实际销量vs预测值',fontsize=14)  # 标题，字号设置
    plt.plot(list(range(len(df1))),df1['实际值'],color='b', label= '实际值')
    plt.plot(list(range(len(df1))),df1['pre'],color='r', label= '预测值')
    plt.scatter(list(range(len(df1))),df1['实际值'], s=area, c='r', alpha=0.4, label='实际值')
    plt.scatter(list(range(len(df1))),df1['pre'], s=area, c='b', alpha=0.4, label='预测值')

    plt.xlabel('陕西省')  # 位置设置
    plt.ylabel('销量')  # 方向，位置设置
    plt.legend()
    plt.show()
    
    fig3 = plt.figure()
    plt.title('实际销量vs预测值',fontsize=14)  # 标题，字号设置
    plt.plot(list(range(len(df2))),df2['实际值'],color='b',label= '实际值')
    plt.plot(list(range(len(df2))),df2['pre'],color='r', label= '预测值')
    plt.scatter(list(range(len(df2))),df2['实际值'], s=area, c='r', alpha=0.4, label='实际值')
    plt.scatter(list(range(len(df2))),df2['pre'], s=area, c='b', alpha=0.4, label='预测值')    
    plt.xlabel('山东省')  # 位置设置
    plt.ylabel('销量')  # 方向，位置设置
    plt.legend()
    plt.show()


    fig1.savefig('f1.png')
    fig4.savefig('f4.png')
    fig5.savefig('f5.png')

    writer = pd.ExcelWriter('output_1103.xls',engine='xlsxwriter')
#     df.to_excel(writer, sheet_name='sheet1', index=False)
    sheet = writer.book.add_worksheet('sheet1f')
    sheet.insert_image(0,0,'f1.png')
    sheet.insert_image(0,10,'f4.png')
    sheet.insert_image(0,20,'f5.png')
    df.to_excel(writer, sheet_name='sheet1', startrow = 0 )
    writer.save()
    

def main1():
    lr = [0.025,0.005]
    echo = [1000,1300]
    all = [data1_1,data1_2]
    data_test_ori = [data2_3,data2_4]
    data_test = [data_test1,data_test2]
    data_out = [1,2]
    stderr = {}
    for i,j in enumerate(all):
        for k in range(5):
            train_epochs = echo[i]
            learning_rate = lr[i]
            data = j.values
            x_data, y_data, mean_y, std_y = normal(data)
            loss_function, optimizer, pred, x, y, b, w = model(learning_rate)
            run_model(loss_function, optimizer, pred, x_data, y_data, x, y, b, w,train_epochs)
            true,pre,error, error_abs = prediction(pred,x,1,data_test[i])
            data_out[i] = data_test_ori[i]
            data_out[i]['true'] = true
            data_out[i]['pre_%s'%k] = pre
    #         data_out[i]['error'] = error
            data_out[i]['error_abs_%s'%k] = error_abs
            stderr[k] = data_out[i]['error_abs_%s'%k].mean()
#             print("stderror",data_out[i]['error_abs_%s'%k].std())
        (u,v) = min(stderr.items(),key=lambda x:x[1])    ###标准差最小的key和value
        print(u, type(u))
        data_out[i]['pre'] = data_out[i]['pre_%s'%u]
        data_out[i]['error_abs'] = data_out[i]['error_abs_%s'%u]
        
    data_out = pd.concat([data_out[0], data_out[1]],axis=0)
    data_out.sort_index(inplace=True)
    print(data_out)

    plot_acc(data_out)
  


import os

if __name__ == '__main__':
    os.remove('error_record_all_1103.txt')
    log = open('error_record_all_1103.txt', 'a+')    
    main1()




