# coding=utf=8


from flask import Flask , render_template , request

import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from sklearn import  linear_model , pipeline , preprocessing

data_x = [1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10]
data_y = [2730 , 4537 , 5997 , 7736 , 9720 , 14411 , 17238 , 20471 , 24363 , 28060]
predict_x = [10 , 11 , 12 , 13 , 14 , 15]
real_y = [28060 , 31211 , 34598 , 37251 , 40235 , 42708]


def into_2D (lists):
    result = []
    for i in lists:
        new = []
        new.append(i)
        result.append(new)
    return result


app = Flask(__name__)


def prediction (predict_x , model_number):
    if model_number == '1' or model_number == None:  # 线性回归
        model = linear_model.LinearRegression()
    elif model_number == '2':  # 贝叶斯岭回归
        model = linear_model.BayesianRidge()
    elif model_number == '3':  # 随机抽样一致性算法
        model = linear_model.RANSACRegressor()
    elif model_number == '4':
        model = pipeline.make_pipeline(
            preprocessing.PolynomialFeatures(3) ,  # 多项式特征拓展器
            linear_model.LinearRegression()  # 线性回归器
        )
    model.fit(np.array(into_2D(data_x)) , np.array(into_2D(data_y)))
    predict_y = model.predict(np.array(into_2D(predict_x)))
    print(predict_y)
    return predict_y


@app.route('/' , methods = ['GET'])
def Main ():
    get_type = request.args.get('type')
    # matplotlib.use('Agg')  # 不出现画图的框
    # 这段正常画图
    plt.axis([1 , 15 , 0 , 50000])  # [xmin,xmax,ymin,ymax]对应轴的范围
    plt.title('Data')  # 图名
    plt.plot(data_x , data_y , 'k')
    plt.plot(predict_x , real_y , 'k')
    plt.plot(predict_x , prediction(predict_x , get_type) , 'r')
    # -----------
    # 转成图片的步骤
    sio = BytesIO()
    plt.savefig(sio , format = 'png')
    data = base64.encodebytes(sio.getvalue()).decode()
    html = '''
           <html>
               <body>
                   <img src="data:image/png;base64,{}" />
               </body>
            <html>
        '''
    plt.close()
    # 记得关闭，不然画出来的图是重复的
    return html.format(data)
    # format的作用是将data填入{}


if __name__ == '__main__':
    app.run()
