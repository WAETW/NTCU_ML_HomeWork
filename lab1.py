import numpy as np

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

def sigmoid(x):
    return 1.0/(1.0*np.exp(-x))

def sigmoid_derivate(x):
    return np.multiply(x, 1.0 - x)

def add_layer(input_dim, output_dim):
    return np.random.rand(input_dim, output_dim)

def train_network(inputs,labels, epoch , lr):   
    w0 = add_layer(2,4)
    w1 = add_layer(4,4)
    w2 = add_layer(4,1)
    for epoch in range(epoch):
        input_layer = inputs
        hiden_layer1 = sigmoid(np.dot(input_layer, w0))
        hiden_layer2 = sigmoid(np.dot(hiden_layer1, w1))
        output_layer = sigmoid(np.dot(hiden_layer2, w2))
        
        output_layer_error = labels - output_layer
        output_layer_delta = output_layer_error * sigmoid_derivate(output_layer)

        hiden_layer2_error = output_layer_delta.dot(w2.T)
        hiden_layer2_delta = hiden_layer2_error * sigmoid_derivate(hiden_layer2)
        
        hiden_layer1_error = hiden_layer2_delta.dot(w1.T)
        hiden_layer1_delta = hiden_layer1_error * sigmoid_derivate(hiden_layer1)
        
        w2 += hiden_layer2.T.dot(output_layer_delta) * lr
        w1 += hiden_layer1.T.dot(hiden_layer2_delta) * lr
        w0 += input_layer.T.dot(hiden_layer1_delta) * lr
    return w0, w1, w2

if __name__ == '__main__':
    inputs, labels = generate_XOR_easy()
    print(train_network(inputs, labels, 10000, 0.5))