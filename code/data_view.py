import numpy as np
import pandas as pd
import seaborn as sns

def get_data_from_file():
    train_set = pd.read_csv('../data/titanic/train.csv')
    test_set = pd.read_csv("../data/titanic/test.csv")
    return train_set,test_set

def get_information_from_dataframe(df):
    print("数据集大小：",df.shape)

def age_survived(data_set):
    #Age与Survived
    age_facet = sns.FacetGrid(data_set,hue='Survived',aspect=3)
    age_facet.map(sns.kdeplot,'Age',shade=True)
    age_facet.set(xlim=(0,data_set['Age'].max()))
    age_facet.add_legend()

if __name__ == "__main__":
    train_set,test_set = get_data_from_file()
    age_survived(train_set)
    # print("训练集信息：")
    # get_information_from_dataframe(train_set)
    # print("测试集信息：")
    # get_information_from_dataframe(test_set)



