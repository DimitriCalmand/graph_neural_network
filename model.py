import numpy as np
import layer
import numpy_graph as npg
class Sequential:
    
    def __init__(self, layers=None):
        #Add the feature if layer is not none like sequencial or a list of layer
        self.layers = list()
    def add(self, layer):
        self.layers.append(layer)
    def compile(self, loss_function):
        self.loss_function = loss_function
        npg.IS_BUILDING = True
        shape = self.layers[0].input_shape
        X = npg.ones((2,)+shape)
        X.input_name = "input" 
        for layer in self.layers:
            layer.build(X)
            X = layer.call(X)
            shape = X.shape[1:]
        id_output = npg.graph.operations[-1][3]
        npg.graph.dico[id_output].output_name = "output"
        y = npg.ones(X.shape)
        y.input_name = "output"
        loss_function(X, y)
        id_loss = npg.graph.operations[-1][3]
        npg.graph.dico[id_loss].output_name = "loss"
        npg.IS_BUILDING = False
        return None
    def predict_proba(self, X):
        y = npg.g_array(np.array([1]))
        predicted = npg.graph.launch_operation({"input" : X, "output": y})
        return predicted["output"]
    def binarize_predictions (self, predicted_array):
        if predicted_array.shape[-1] > 1:
            res = np.argmax(predicted_array, axis = -1)
        else:
            res = np.zeros_like(predicted_array)
            res[predicted_array > 0.5] = 1
            res = np.squeeze(res)
        return res

    def predict(self, X):
        predicted_array = self.predict_proba(X).array
        res = self.binarize_predictions(predicted_array)
        res = npg.g_array(res)
        return res
    def accuracy(self, predicted, y):
        pred = self.binarize_predictions(predicted.array)
        true = self.binarize_predictions(y.array)
        good = pred == true
        res = np.sum(good) / np.size(pred)
        return npg.g_array(res)
        
         

    def apply_gradient(self, gradients):
        for (g_array, gradient) in gradients:
            if g_array.is_bias :
                gradient = gradient.sum(axis = 0, keepdims = True)
            g_array.array -= gradient.array 
    def train_step(self, X, y):
        predicted = npg.graph.launch_operation({"input" : X, "output": y})
        loss = predicted['loss']
        predict = predicted['output']
        output = npg.ones(loss.shape)
        gradients = npg.graph.derivative_operation(output)
        self.apply_gradient(gradients)
        return loss, predict
    def fit(self, X, y, epoch, batch = 64, verbose = 1):
        print(X[1])
        for i in range(epoch):
            general_loss = 0 
            general_acc = 0
            nb_iter = X.shape[0]//batch
            for j in range(nb_iter):
                X_batch = X[j:j+batch]
                y_batch = y[j:j+batch]
                loss, predict = self.train_step(X_batch, y_batch)
                general_loss += loss.array[0]
                general_acc += self.accuracy(predict, y_batch)
            if verbose != 0 and i%verbose == 0:
                str_loss = f"loss = {round(general_loss/nb_iter, 5)}"
                str_acc = f"acc = {general_acc/nb_iter}"
                print(f"epoch:{i}/{epoch} | {str_loss} | {str_acc}")




