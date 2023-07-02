

class Node:

    def __init__(self):

        self.data = # single data item
        self.weights_key = # the weights of the key linear layer
        self.weights_value = # the weights of the value linear layer
        self.weights_query = # ...

    def keys(self):
        # What do I have - what do I know about the data that I have?
        return self.weights_key @ self.data 

    def query(self):
        # What am I looking for ?
        return self.weights_query @ self.data 
    
    def value(self):
        # What do I broadcast? What do I tell people.
        return self.weights_value @ self.data 