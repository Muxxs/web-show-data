#coding=utf-8

def get_data_from_csv(data,main_cou):
    file=open("data/"+data).read()
    num=file.split("\n")[0].split(",").index(main_cou)
    print(num)
    y_list=[]
    x_list=[]
    x_predict=[]
    count=1
    for i in file.split("\n")[1:]:
        y_list.append(int(i.split(",")[num]))
        x_list.append(count)
        count+=1
    for x in range(0,6):
        x_predict.append(count+x)
    return x_list,y_list,x_predict
