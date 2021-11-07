import pandas as pd
data1 = pd.read_excel('大中马力误差记录/大马力.xls',sheet_name = 'sheet1', usecols=[1,2,8,25,26],header = 0, encoding='utf-8')

data2 = pd.read_excel('大中马力误差记录/中马力.xls',sheet_name = 'sheet1', usecols=[1,2,8,20,21],header = 0, encoding='utf-8')

df = pd.concat([data1, data2], axis = 0)
print(df)

def split_range(df):
    t1 = []
    t2 = []
    o_abs1 = []
    tr_sum1 = []
    pre_sum1 = []
    o_abs2 = []
    tr_sum2 = []
    pre_sum2 = []

    engin1= []
    engin2 = []
    
    for i in range(7):
        if i <1 :
            t1.append(df[(df['马力'] >=25+10*i) & (df['马力'] <40)])
            engin1.append("25-40")
        else:
            t1.append(df[(df['马力'] >=30+10*i) & (df['马力'] <(i+1)*10+30)])
            engin1.append("%s-%s"%(30+10*i,(i+1)*10+30))
        
        o1 = t1[i]['实际值'].sum()
        o2 = t1[i]['pre'].sum()
        o_abs1.append((o2-o1)/o1)
        tr_sum1.append(o1)
        pre_sum1.append(o2)
    
    
    if len(df[df['马力'] >= 240]) >= 1:
        iter = 8
    else:
        iter = 7
    for i in range(iter):
        if i <7 :
            t2.append(df[(df['马力'] >=100+20*i) & (df['马力'] <100+20*(i+1))])
            engin2.append("%s-%s"%(100+20*i,100+20*(i+1)))
        else:
            if len(df[df['马力'] >= 240]) >= 1:
                t2.append(df[(df['马力']>= 240) ])
                engin2.append(">240")
    

        o1 = t2[i]['实际值'].sum()
        o2 = t2[i]['pre'].sum()
        o_abs2.append((o2-o1)/o1)
        tr_sum2.append(o1)
        pre_sum2.append(o2)
    tr_sum = tr_sum1 + tr_sum2
    pre_sum = pre_sum1 + pre_sum2
    o_abs = o_abs1 + o_abs2
    engin = engin1 + engin2
    

    return tr_sum,pre_sum,o_abs,engin


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # 能快速读取常规大小的文件。Pandas能提供高性能、易用的数据结构和数据分析工具
from sklearn.utils import shuffle  # 随机打乱工具，将原有序列打乱，返回一个全新的顺序错乱的值
from collections import OrderedDict
import xlwt
def plot_acc(df):

    ##数据分为西安和山东两组，分别输出世纪指与预测值的曲线对比图
    df1 = df[df['省份'] == '陕西']
    df2 = df[df['省份'] == '山东']
    tr_sum1, pre_sum1, o_abs1,engin1 = split_range(df1)
    tr_sum2, pre_sum2, o_abs2, engin2 = split_range(df2)
    engin = [engin1,engin2]
#     print("hahahah", o_abs1)
        
    colors1 = '#00CED1' #点的颜色
    colors2 = '#DC143C'
    area = np.pi * 4**2  # 点面积
#     engin = ['100-120','120-140','140-160','160-180','180-200','200-220','220-240', '>240']

    plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
    fig1 = plt.figure()
    plt.title('各个马力区间绝误差率',fontsize=14)  # 标题，字号设置
    plt.plot(engin1,o_abs1,color='r', label= '陕西省: average=%s'%(sum(o_abs1)/len(o_abs1)))
    plt.plot(engin2,o_abs2,color='b', label= '山东省: average=%s'%(sum(o_abs2)/len(o_abs2)))
    plt.scatter(engin1,o_abs1, s=area, c='r', alpha=0.4, label='陕西省')
    plt.scatter(engin2,o_abs2, s=area, c='b', alpha=0.4, label='山东省')
    print("陕西省误差：")
    for i,j in zip(engin1,o_abs1):
        print("%s: %s"%(i,j))
        
        

    plt.ylim([-5, 5])
    plt.xlabel('马力区间')  # 位置设置
    plt.ylabel('误差率')  # 方向，位置设置
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
 
    fig4 = plt.figure()
    plt.title('陕西各马力区间实际值vs预测值',fontsize=14)  # 标题，字号设置
    plt.plot(engin1,tr_sum1,color='r', label= '实际值')
    plt.plot(engin1,pre_sum1,color='b', label= '预测值')
    plt.scatter(engin1,tr_sum1, s=area, c='r', alpha=0.4, label='实际值')
    plt.scatter(engin1,pre_sum1, s=area, c='b', alpha=0.4, label='预测值')

    plt.xlabel('马力区间')  # 位置设置
    plt.ylabel('销量')  # 方向，位置设置
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

    
    fig5 = plt.figure()
    plt.title('山东各个马力区间绝实际值vs预测值',fontsize=14)  # 标题，字号设置
    plt.plot(engin2,tr_sum2,color='r', label= '实际值')
    plt.plot(engin2,pre_sum2,color='b', label= '预测值')
    plt.scatter(engin2,tr_sum2, s=area, c='r', alpha=0.4, label='实际值')
    plt.scatter(engin2,pre_sum2, s=area, c='b', alpha=0.4, label='预测值')

    plt.xlabel('马力区间')  # 位置设置
    plt.ylabel('销量')  # 方向，位置设置
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

    fig1.savefig('大马力误差率.png')
    fig4.savefig('大马力陕西预测.png')
    fig5.savefig('大马力山东预测.png')
    writer = pd.ExcelWriter('大马力最优解结算结果-1106.xls',engine='xlsxwriter')
    sheet = writer.book.add_worksheet('sheet1f')
    sheet.insert_image(0,0,'大马力误差率.png')
    sheet.insert_image(0,10,'大马力陕西预测.png')
    sheet.insert_image(0,20,'大马力山东预测.png')
    df.to_excel(writer, sheet_name='sheet1', startrow = 0 )
    writer.save()

plot_acc(df)
