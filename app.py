# coding=utf=8
'''
@Email: muxxs@foxmail.com
@Auther: Muxxs
'''

from flask import Flask , render_template , request
from model import deal_csv
import matplotlib.pyplot as plt
from io import BytesIO
import base64 , os
import numpy as np
from sklearn import linear_model , pipeline , preprocessing

ALLOWED_EXTENSIONS = set(['csv' , 'png'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "data/"


def choose_data (data_type):
    if data_type == '1':
        data_x = [1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15]
        data_y = [2730 , 4537 , 5997 , 7736 , 9720 , 14411 , 17238 , 20471 , 24363 , 28060 , 31211 , 34598 , 37251 ,
                  40235 , 42708]
        predict_x = [15 , 16 , 17 , 18 , 19 , 20]
        return data_x , data_y , predict_x


def into_2D (lists):
    result = []
    for i in lists:
        new = []
        new.append(i)
        result.append(new)
    return result


app = Flask(__name__)


def prediction (data_x , data_y , predict_x , model_number):
    if model_number == '1':  # 线性回归
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
    elif model_number == '5':
        model = pipeline.make_pipeline(
            preprocessing.PolynomialFeatures(2) ,  # 多项式特征拓展器
            linear_model.LinearRegression()  # 线性回归器
        )
    model.fit(np.array(into_2D(data_x)) , np.array(into_2D(data_y)))
    predict_y = model.predict(np.array(into_2D(predict_x)))
    return predict_y


def allowed_file (filename):
    return '.' in filename and \
           filename.rsplit('.' , 1)[1] in ALLOWED_EXTENSIONS


@app.route('/' , methods = ['GET' , 'POST'])
def Main ():
    try:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join("data" , filename))
        main_cou = request.form.get("main_cou")
        get_type = request.form.get('type')
        if get_type == None:
            get_type = '1'
        data_x , data_y , predict_x = deal_csv.get_data_from_csv(filename , main_cou)
    except Exception as e:
        print(e)
        get_type = request.form.get('type')
        get_data = request.form.get('data')
        if get_type == None:
            get_type = '1'
        if get_data == None:
            get_data = '1'
        data_x , data_y , predict_x = choose_data(get_data)
    # matplotlib.use('Agg')  # 不出现画图的框
    # 这段正常画图



    predictions = prediction(data_x , data_y , predict_x , get_type)
    print(data_x)
    print(data_y)
    print(predict_x)
    print(predictions)
    plt.axis(
        [min(data_x) ,
         max([max(data_x) , max(predict_x)]) ,
         min([min(data_y) , min(predictions)]) ,
         max([max(data_y) , max(predictions)])
         ]
    )  # [xmin,xmax,ymin,ymax]对应轴的范围
    plt.title('Data')  # 图名
    plt.plot(data_x , data_y , 'k')
    plt.plot(predict_x , predictions , 'r')
    # -----------
    # 转成图片的步骤
    sio = BytesIO()
    plt.savefig(sio , format = 'png')
    data = base64.encodebytes(sio.getvalue()).decode()
    html = open("template/index.html").read().replace('option value="' + get_type + '"' ,
                                                      'option value="' + get_type + '" selected="selected"')
    plt.close()
    # 记得关闭，不然画出来的图是重复的
    return html.format(data)
    # format的作用是将data填入{}


if __name__ == '__main__':
    app.run()
