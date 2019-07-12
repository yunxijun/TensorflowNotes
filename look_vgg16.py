import os
import numpy as np


class Vgg16():
    def __init__(self, vgg16_path=None):
        if vgg16_path is None:
            vgg16_path = os.path.join(os.getcwd(), "vgg_weight/vgg16.npy")
            self.data_dict = np.load(vgg16_path, encoding='latin1').item()
            print(self.data_dict)
        for x in self.data_dict:
            print(x)


def main():
    return Vgg16()


if __name__ == "__main__":
    main()