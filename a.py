# from tensorflow.keras import Sequential
# from tensorflow.keras import layers
# model = Sequential()
# model.add(layers.Conv2D(10,(3,3), activation='relu', strides=(1,1),input_shape=(256,256,3)))
# model.add(layers.Conv2D(30, (11,11),activation='relu'))
# model.add(layers.GlobalAvgPool2D())
# model.add(layers.Dense(1000, activation='relu'))
# model.summary()

class A:
    def __init__(self):
        self.s = 1
        # self.d = None
    def d2(self):
        self.d+=1
        return self.d

if __name__=="__main__":
    a = A()
    a.d = 1
    print(a.d2())