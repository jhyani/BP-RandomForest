#BP_script
- 大马力和中马力分别基于学习率与训练次数循环125次，找到平均误差最小的那一组参数
#script:
```def main1():
    lr = [0.005,0.01,0.015,0.02,0.025]
    echo = [400,700,1000,1300,1600]
    all = [data1_1,data1_2]
    data_test_ori = [data2_3,data2_4]
    data_test = [data_test1,data_test2]
    data_out = {}
    error_out = {}
    each = 0    
    for m in lr:
        for n in echo:
            train_epochs = n
            learning_rate = m 
            repeat = 0
            error_out[each] = pd.DataFrame()
            range_err = pd.DataFrame()
            range_err['马力区间'] = ['100-120','120-140','140-160','160-180','180-200','200-220','220-240','>240']
            for k in range(5):
                data = all[1].values
                x_data, y_data, mean_y, std_y = normal(data)
                loss_function, optimizer, pred, x, y, b, w = model(learning_rate)
                run_model(loss_function, optimizer, pred, x_data, y_data, x, y, b, w,train_epochs)
                true,pre,error, error_abs = prediction(pred,x,1,data_test[1])```