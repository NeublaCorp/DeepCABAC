from cgi import test
from math import ceil, floor
import pandas as pd
import numpy as np
import re
import onnx
from onnx import numpy_helper
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

import deepCABAC
import pdb

def deep_CABAC_ec_fp32tobyte(x):
    addSteps = 256
    interv = 0.22
    _labmda = 0
    
    weightAddHalfInterv = ( np.abs( x.flatten() ) + interv / 2. )
    maxIdx              = np.argmax( weightAddHalfInterv )
    weightsRange = 2*weightAddHalfInterv[ maxIdx ]

    nSteps       = np.ceil( weightsRange / interv )
    nSteps      += addSteps
    stepsize = weightsRange / nSteps

    encoder = deepCABAC.Encoder()
    encoder.encodeWeightsRD(x, interv, stepsize, _lambda)
    stream = encoder.finish().tobytes()
    return stream

def deep_CABAC_dc_bytetoint8(x):
    decoder = deepCABAC.Decoder()
    decoder.getStream(np.frombuffer(x, dtype=np.uint8))

    param = decoder.decodeWeights_wstepsize()
    q_weight_data = param[0]
    scale = np.float32(param[1])
    zp = np.int8(0)
    decoder.finish()
    return q_weight_data, scale, zp

def deep_CABAC(x):
    stream = deep_CABAC_ec_fp32tobyte(x)    
    stream_len = len(stream)
    weight_int8, scale, zp = deep_CABAC_dc_bytetoint8(stream)
    return stream_len, weight_int8, scale, zp

def getComprRatio(stream_len, q_weight):
    s_numBit = (stream_len) * 8
    q_numBit = q_weight.size*q_weight.flatten()[0].nbytes * 8
    comprRatio = s_numBit / q_numBit
    return comprRatio

def reshapeForWeight(t:np.ndarray, tZP:np.ndarray, format="MCSRC2M32"):
    shape_t = t.shape #[M,C,S,R]
    mSize = shape_t[0]
    cSize = shape_t[1]
    sSize = shape_t[2]
    rSize = shape_t[3]
    layerWiseQuant = tZP.size == 1
    if(format=="MCSRC2M32"):
        new_t = np.zeros((ceil(mSize/32),ceil(cSize/2),sSize,rSize,2,32),dtype=np.int8)
    elif(format=="MCSRC1M32"):
        new_t = np.zeros((ceil(mSize/32),cSize,sSize,rSize,32),dtype=np.int8)
    else:
        print(f'{format} is not supported for weight')
        sys.exit(-1)
    for m in range(mSize):
        for c in range(cSize):
            for s in range(sSize):
                for r in range(rSize):
                    try:
                        if(layerWiseQuant): new_dataZP = tZP.item()
                        else: new_dataZP = tZP[m] 
                        new_data = t[m][c][s][r] - new_dataZP
                        if(format=="MCSRC2M32"):
                            new_t[floor(m/32)][floor(c/2)][s][r][c%2][m%32] = new_data
                            # print(f'{m},{c},{s},{r} => {floor(m/32)},{floor(c/2)},{s},{r},{c%2},{m%32}')
                        else:
                            new_t[floor(m/32)][c][s][r][m%32] = new_data
                    except:
                        print(f'Error to mapping {m},{c},{s},{r} => {floor(m/32)},{floor(c/2)},{s},{r},{m%2},{m%32}')
                        sys.exit(-1)
    return new_t

parser = argparse.ArgumentParser(description='Test DeepCABAC with quatized model')
parser.add_argument('--model', required=True, help='quantized model')
parser.add_argument('--target_layer_num', required=False, default=0, help='the number of tes target layers, Defualt=0(all layres of model)')
parser.add_argument('--target_start_layer', required=False, default=0, help='start number of test target layers.')
args = parser.parse_args()

modelPath = args.model
modelName = Path(modelPath).stem
# comprName = "EBPC"
comprName = "dCABAC"
#for BPC
wordwidth = 8; chunkSize = 8
#for CABAC
interv=0.1; stepsize=15; _lambda=0.

onnx_model = onnx.load(modelPath)
conv2weightName_dict = {}
weight_num = 0
for t in onnx_model.graph.node:
    if re.search("Conv",t.op_type):
        weight_num += 1
        weight_name = t.input[1] # fp32 weight
        conv2weightName_dict[weight_name] = t.name

test_num = args.target_layer_num
target_layer = args.target_start_layer
current_layer = 0

compression_ratio_sum = 0
test_layer_num = 0
result_df = pd.DataFrame(columns=["DataName","CompRation","zeroRatio"])

f, axes = plt.subplots(weight_num, figsize=(10,2*weight_num))
plt.subplots_adjust(wspace = 0.3, hspace = 1)
zero_count = 0
weight_count = 0

for t in onnx_model.graph.initializer:
    if t.name in conv2weightName_dict.keys():
        if current_layer < target_layer:
            continue
        nodeName = conv2weightName_dict[t.name]
        test_num = test_num - 1
        weight_data_fp32 = numpy_helper.to_array(t) #fp32

        #print('tensor dim - {}'.format(weight_data_fp32.shape))

        # deepcabac (fp32 -> byte -> int8)
        stream_len, weight_data, scale, weightZP_data  = deep_CABAC(weight_data_fp32)
 
        if weight_data.shape[1] % 2:
            weight_data_nb = reshapeForWeight(weight_data,weightZP_data,format="MCSRC2M32") #int8
        else:
            weight_data_nb = reshapeForWeight(weight_data,weightZP_data,format="MCSRC1M32") #int8

        zero_count += np.count_nonzero(weight_data==weightZP_data)
        weight_count += weight_data.size
        axes[current_layer].hist(weight_data_nb.flatten(),bins=30)
        axes[current_layer].set_title(f'{conv2weightName_dict[t.name]}.{t.name}', fontsize = 10)
        current_layer = current_layer+1

        compRatio = getComprRatio(stream_len, weight_data_nb)
        compression_ratio_sum += compRatio
        test_layer_num += 1
        print(f'Get weight from {conv2weightName_dict[t.name][0]}.{t.name}, CompRation = {compRatio}')
        result_df = result_df.append({'DataName':f'{conv2weightName_dict[t.name][0]}.{t.name}'
                                    , "CompRation": compRatio
                                    , "zeroRatio": f'{np.count_nonzero(weight_data==0)/weight_data.size:.4f}'
                                    }, ignore_index=True)
        if test_num == 0:
            break
plt.savefig(f'{modelName}.weight.png')
result_df.to_csv(f'{comprName}.{modelName}.compRatio.csv')
print(f'{comprName}: Avg compression ratio = {compression_ratio_sum:.4f} / {test_layer_num} = {compression_ratio_sum/test_layer_num:.4f}')
print(f'{comprName}: Avg zero ratio        = {zero_count} / {weight_count} = {zero_count/weight_count:.4f}')

