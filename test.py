import mxnet as mx
import numpy as np



def test():
    np_arr_1 = np.ones([12,5,5],dtype=int)
    np_arr_2 = np.ones([12,1,1],dtype=int)

    np.broadcast(np_arr_1,np_arr_2)
    print(np_arr_1 + np_arr_2)

    arr = mx.nd.array(np_arr_1)
    arr_2 = mx.nd.array(np_arr_2)

    arr3 = mx.nd.broadcast_mul(arr,arr_2)

    print(arr3)
    print("wo shi")


if __name__ == '__main__':
    test()
