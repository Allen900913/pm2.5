import pandas as pd
import numpy as np

# 資料預處理
def dataProcess(df):
    x_list, y_list = [], []
    # 將指定元素替換為數字，將空數據替換為 0
    df = df.replace(['NR'], [0.0])
    # 使用 astype() 轉換 array 中元素的數據類型
    array = np.array(df).astype(float)
    # 將數據集拆分為多個小矩陣
    for i in range(0, 4320, 18):
        for j in range(15):
            input = array[i+9, j:j+9]  # 單行的9個元素
            label = array[i+9, j+9]  # 第10行的 PM2.5 值
            x_list.append(input)
            y_list.append(label)
    x = np.array(x_list)
    y = np.array(y_list)
    
    return x, y, array

# 更新參數並訓練模型
def train(x_train, y_train, epoch):
    bias = 0  # 偏置值初始化
    weights = np.ones(9)  # 權重初始化，與 input 的維度匹配
    learning_rate = 1  # 初始學習率
    reg_rate = 0.001  # 正則項系數
    b_grad2_sum = 0  # 儲存bias的梯度平方和
    w_grad2_sum = np.zeros(9)  # 儲存weight的梯度平方和

    for i in range(epoch):
        b_grad = 0
        w_grad = np.zeros(9)
        
        # 計算在所有數據上的梯度
        for j in range(3200):
            error = y_train[j] - weights.dot(x_train[j]) - bias
            b_grad += -error
            w_grad += -error * x_train[j]

        # 計算平均梯度
        b_grad /= 3200
        w_grad /= 3200
        # 加上正則項的梯度
        w_grad += reg_rate * weights

        # 使用 Adagrad 優化算法
        b_grad2_sum += b_grad**2
        w_grad2_sum += w_grad**2
        # 更新權重與偏置
        bias -= (learning_rate / (b_grad2_sum**0.5)) * b_grad
        weights -= (learning_rate / (w_grad2_sum**0.5)) * w_grad

        # 每訓練 200 輪，輸出訓練集上的損失
        if i % 200 == 0:
            loss = np.mean((y_train - x_train.dot(weights) - bias) ** 2)
            print(f'訓練 {(i+200)} 輪後，訓練數據的損失為:', loss)

    return weights, bias

# 驗證模型效果
def validate(x_test, y_test, weights, bias):
    loss = np.mean((y_test - x_test.dot(weights) - bias) ** 2)
    return loss

def main():
    # 從 CSV 中讀取數據，若編碼錯誤可使用 encoding = 'big5' 或 'gb18030'
    df = pd.read_csv('train.csv', usecols=range(3, 27), encoding='big5')
    x, y, _ = dataProcess(df)
    
    # 划分訓練集與驗證集
    x_train, y_train = x[:3200], y[:3200]
    x_test, y_test = x[3200:3600], y[3200:3600]
    epoch = 2000  # 訓練輪數
    
    # 開始訓練
    w, b = train(x_train, y_train, epoch)
    
    # 驗證集上的效果
    loss = validate(x_test, y_test, w, b)
    print('驗證數據的損失為:', loss)

if __name__ == '__main__':
    main()
