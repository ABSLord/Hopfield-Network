from HopfieldNetwork import Hopfield
from HopfieldNetwork import threshold_function


if __name__ == "__main__":
    nn = Hopfield(100, 100, threshold_function)
    nn.teach()
    nn.detect("./detect/a_star3.jpg", 10)
    print(nn.hamming_distance(save=True))