import numpy as np

graph = None 
IS_BUILDING = False
MAX_ID = 0
NEED_ID = True

class Graph:
    def __init__(self,):
        self.dico = {}
        self.operations = []
    def add_operation(self, symbole:str, value1, value2,  res, **kwargs):
        self.dico[res.id] = res
        if value1.id not in self.dico:
            self.dico[value1.id] = value1
        if value2 is not None and value2.id not in self.dico:
            self.dico[value2.id] = value2
        if value2:
            self.operations.append((symbole, value1.id, value2.id, res.id, kwargs))
        else:
            self.operations.append((symbole, value1.id, None, res.id, kwargs))

    def launch_operation(self, inputs):
        NEED_ID = False
        res = None
        list_res = {}
        for operation in self.operations:
            res = self.select_operation(operation, inputs)
            id_res = operation[3] 
            self.dico[id_res] = res
            if res.output_name is not None:
                list_res[res.output_name] = res
        NEED_ID = True
        return list_res 

    def select_operation(self, operation, inputs):
        symbole, id1, id2, id_res, kwargs = operation
        value1 = self.dico[id1]
        if value1.input_name is not None:
            value1.array = inputs[value1.input_name].array
        if id2 is not None:
            value2 = self.dico[id2]
            if value2.input_name is not None: 
                value2.array = inputs[value2.input_name].array
        if not isinstance(symbole, str):
              res = symbole(value1, **kwargs)  
        else:
            if symbole == '+':
                res = value1 + value2
            elif symbole == '@':
                res = value1 @ value2
            elif symbole == '*':
                res = value1 * value2
            elif symbole == '-':
                res = value1 - value2
            elif symbole == 'neg':
                res = -value1
            elif symbole == '/':
                res = value1 / value2
            elif symbole == '**':
                res = value1 ** value2
            elif symbole == '[]':
                res = value1[kwargs["index"]]
            elif symbole == '=':
                value1[kwargs["index"]] = kwargs["value"]
                res = value1
            elif symbole == 'log':
                res = log(value1)
            elif symbole == 'e':
                res = exp(value1)
        res.id = id_res
        res.input_name = self.dico[id_res].input_name
        res.output_name = self.dico[id_res].output_name
        res.is_bias = self.dico[id_res].is_bias
        return res
    def derivative_operation(self, output):
        self.debug = False
        list_gradient = []
        list_id = {}
        NEED_ID = False
        self.dico_derivative = {self.operations[-1][3] : output}
        for operation in reversed(self.operations):
            self.select_derivative_operation(operation, list_gradient, list_id)
        NEED_ID = True
        return list_gradient 
    def select_derivative_operation(self, operation, list_gradient, list_id):
        symbole, id1, id2, id_res, kwargs = operation
        value1 = self.dico[id1]
        if id1 not in self.dico_derivative:
            self.dico_derivative[id1] = zeros_like(value1)
        res = self.dico_derivative[id_res]
        if id2 is not None:
            value2 = self.dico[id2]
            if id2 not in self.dico_derivative:
                self.dico_derivative[id2] = zeros_like(value2)
        #print("res = ", res, f'\n {symbole},{value1} \n', self.dico[id_res])
        if self.dico[id_res].output_name == 'output':
            self.debug = True
            pass
        if not isinstance(symbole, str):
            self.dico_derivative[id1] = symbole(value1, derivative = True, **kwargs) * res
        else:
            if symbole == '+':
                self.dico_derivative[id1] += res 
                self.dico_derivative[id2] += res 
            elif symbole == '@':
                self.dico_derivative[id1] += res @ value2.T
                self.dico_derivative[id2] += value1.T @ res
            elif symbole == '*':
                self.dico_derivative[id1] += value2 * res
                self.dico_derivative[id2] += value1 * res
            elif symbole == '/':
                self.dico_derivative[id1] += (res / value2)
                self.dico_derivative[id2] += -(res * value1) / (value2 ** 2)
            elif symbole == '-':
                self.dico_derivative[id1] += res 
                self.dico_derivative[id2] += -res 
            elif symbole == 'neg':
                self.dico_derivative[id1] += -res
            elif symbole == '**':
                self.dico_derivative[id1] += res * value2 * (value1 ** (value2- 1))
                self.dico_derivative[id2] += res * log(value1) * self.dico[id_res]
            elif symbole == '[]':
                index = kwargs["index"]
                self.dico_derivative[id1][index] += res 
            elif symbole == '=':
                index = kwargs["index"]
                self.dico_derivative[id1] += res
                self.dico_derivative[id1][index] = 0
            elif symbole == 'log':
                self.dico_derivative[id1] += res / value1
            elif symbole == 'e':
                self.dico_derivative[id1] += res * exp(value1)
        if self.debug:
            #print("operation = ", symbole)
            pass
            #print(self.dico_derivative[id1])
        if value1.trainable:
            if id1 in list_id:
                list_gradient[list_id[id1]][1] = self.dico_derivative[id1]
            else:
                list_id[id1] = len(list_gradient)
                list_gradient.append([value1, self.dico_derivative[id1]])
        if id2 is not None and  value2.trainable:
            if id2 in list_id:
                list_gradient[list_id[id2]][1] = self.dico_derivative[id2]
            else:
                list_id[id2] = len(list_gradient)
                list_gradient.append([value2, self.dico_derivative[id2]])



class g_array :
    def __init__(self, array, trainable = False, input_name = None, output_name = None, is_bias = False):
        global MAX_ID, graph, IS_BUILDING, NEED_ID
        if type(array) == type(np.array([1])) :
            self.array = array
        else:
            self.array = np.array(array)
        if NEED_ID:
            self.id = MAX_ID
            MAX_ID += 1
        else:
            self.id = None 
        self.trainable = trainable
        self.input_name = input_name
        self.output_name = output_name
        self.is_bias = is_bias 
    @property
    def T(self):
        return transpose(self)
    @property
    def shape(self):
        return self.array.shape
    @property
    def dtype(self):
        return self.array.dtype
    def __str__(self):
        return self.array.__str__()

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            start, stop = indices.start, indices.stop
            result = self.array[start:stop]
        else:
            result = (self.array[indices])
        result = g_array(result)
        if IS_BUILDING :
            graph.add_operation("[]", self, None, result, index=indices)
        return result
    
    def __setitem__(self, index, value):
        if isinstance(value, g_array):
            self.array[index] = value.array
        else:
            self.array[index] = value
        if IS_BUILDING :
            graph.add_operation("=", self, None, None, index=index, value=value)

        
    def __add__(self, other):
        if isinstance(other, g_array):
            res = self.array + other.array
        elif isinstance(other, np.ndarray):
            res = self.array + other
            other = g_array(other)
        else:
            res = self.array + np.array(other)
            other = g_array(np.array(other))
        res = g_array(res)
        if IS_BUILDING :
            graph.add_operation("+", self, other, res)
        return res
    def __radd__(self, other):
        return self.__add__(other)
    def __mul__(self, other):
        if isinstance(other, g_array):
            res = self.array * other.array
        elif isinstance(other, np.ndarray):
            res = self.array * other
            other = g_array(other)
        else:
            res = self.array * np.array(other)
            other = g_array(np.array(other))
        res = g_array(res)
        if IS_BUILDING :
            graph.add_operation("*", self, other, res)
        return res
    def __truediv__(self, other):
        if isinstance(other, g_array):
            res = self.array / other.array
        elif isinstance(other, np.ndarray):
            res = self.array / other
            other = g_array(other)
        else:
            res = self.array / np.array(other)
            other = g_array(np.array(other))
        res = g_array(res)
        if IS_BUILDING :
            graph.add_operation("/", self, other, res)
        return res
    def __rtruediv__(self, other):
        if isinstance(other, g_array):
            res = self.array / other.array
        elif isinstance(other, np.ndarray):
            res = self.array / other
            other = g_array(other)
        else:
            res = self.array / np.array(other)
            other = g_array(np.array(other))
        res = g_array(res)
        if IS_BUILDING:
            graph.add_operation("/", other, self, res)
        return res
    def __sub__(self, other):
        if isinstance(other, g_array):
            res = self.array - other.array
        elif isinstance(other, np.ndarray):
            res = self.array - other
            other = g_array(other)
        else:
            res = self.array - np.array(other)
            other = g_array(np.array(other))
        res = g_array(res)
        if IS_BUILDING :
            graph.add_operation("-", self, other, res)
        return res
    def __rsub__(self, other):
        if isinstance(other, np.ndarray):
            res = other - self.array
            other = g_array(other)
        else:
            res = np.array(other) - self.array
            other = g_array(np.array(other))
        res = g_array(res)
        if IS_BUILDING:
            graph.add_operation("-", other, self, res)
        return res

    def __pow__(self, other):
        if not isinstance(other, g_array):
            other = g_array(np.array([other]))
        res = self.array ** other.array
        res = g_array(res)
        if IS_BUILDING :
            graph.add_operation("**", self, other, res)
        return res
    def __matmul__(self, other):
        new_array = self.array @ other.array
        res = g_array(new_array)
        if IS_BUILDING:
            graph.add_operation("@", self, other, res)
        return res 
    def __neg__(self):
        new_array = -self.array
        res = g_array(new_array)
        if IS_BUILDING:
            graph.add_operation("neg", self, None, res)
        return res
    def mean(self, **kwargs):
        return mean(self, **kwargs)
    def sum(self, **kwargs):
        return sum(self, **kwargs)

def log(garray):
    new_array = np.log(garray.array)
    res = g_array(new_array)
    if IS_BUILDING:
        graph.add_operation("log", garray, None, res)
    return res

def exp(garray):
    new_array = np.exp(garray.array)
    res = g_array(new_array)
    if IS_BUILDING:
        graph.add_operation("e", garray, None, res)
    return res
def check_build(function, garray, res, **kwargs):
    if IS_BUILDING:
        graph.add_operation(function, garray, None, res, **kwargs)
 
def transpose(garray, derivative = False, **kwargs):
    new_array = np.transpose(garray.array, **kwargs)
    res = g_array(new_array)
    check_build(transpose, garray, res, **kwargs)
    return res
def mean(garray, derivative = False, **kwargs):
    if derivative :
        return g_array(np.array([1]))
    res = garray.array.mean(**kwargs)
    res = g_array(res)
    check_build(mean, garray, res, **kwargs)
    return res
def sum(garray, derivative = False, **kwargs):
    res = garray.array.sum(**kwargs)
    res = g_array(res)
    check_build(sum, garray, res, **kwargs)
    return res
def squeeze(garray, derivative = False, **kwargs):
    res = np.squeeze(garray, **kwargs)
    res = g_array(res)
    return res
def zeros_like(garray, **kwargs):
    res = zeros(garray.shape)
    return res


def ones(shape):
    return g_array(np.ones(shape))
def zeros(shape):
    return g_array(np.zeros(shape))
def full(shape, value):
    return g_array(np.full(shape, value))
graph = Graph()

