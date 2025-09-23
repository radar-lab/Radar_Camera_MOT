
from nets.facenet import facenet
from utils.utils import net_flops

if __name__ == "__main__":
    input_shape = [160, 160]
    backbone    = "mobilenet"
    model       = facenet([input_shape[0], input_shape[1], 3], 10575, backbone=backbone, mode="train")

    model.summary()

    net_flops(model, table=False)
    

    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
