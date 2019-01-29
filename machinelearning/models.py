import numpy as np

import backend
import nn

class Model(object):
    """Base model class for the different applications"""
    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            # print "x", x
            # print "y", y
            graph.backprop()
            graph.step(self.learning_rate)

class RegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.075
        layerSize = 50
        # print "batchSize", batchSize
        self.W1 = nn.Variable(1,layerSize)
        self.b1 = nn.Variable(200,layerSize)
        self.W2 = nn.Variable(layerSize,1)
        self.b2 = nn.Variable(1,1)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        # At each iteration, we first calculate a loss that measures how
        # good our network is. The graph keeps track of all operations used
        graph = nn.Graph([self.W1, self.b1, self.W2, self.b2])
        input_x = nn.Input(graph, x)
        xW1 = nn.MatrixMultiply(graph, input_x, self.W1)
        xW1_plusb1 = nn.MatrixVectorAdd(graph, xW1, self.b1)
        afterReLU = nn.ReLU(graph, xW1_plusb1)
        x2W2 = nn.MatrixMultiply(graph, afterReLU, self.W2)
        x2W2_plusb2 = nn.MatrixVectorAdd(graph, x2W2, self.b2)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, x2W2_plusb2, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            output = graph.get_output(x2W2_plusb2)
            # print "Output matrix size:", output.shape
            return output



class OddRegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.075
        layerSize = 200
        self.W1 = nn.Variable(1,layerSize)
        self.b1 = nn.Variable(layerSize)
        self.W2 = nn.Variable(layerSize,1)
        self.b2 = nn.Variable(1)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        # At each iteration, we first calculate a loss that measures how
        # good our network is. The graph keeps track of all operations used
        graph = nn.Graph([self.W1, self.b1, self.W2, self.b2])

        input_x = nn.Input(graph, x)
        neg_x = nn.Input(graph, -1*x)

        xW1 = nn.MatrixMultiply(graph, input_x, self.W1)
        neg_xW1 = nn.MatrixMultiply(graph, neg_x, self.W1)

        xW1_plusb1 = nn.MatrixVectorAdd(graph, xW1, self.b1)
        neg_xW1_plusb1 = nn.MatrixVectorAdd(graph, neg_xW1, self.b1)

        afterReLU = nn.ReLU(graph, xW1_plusb1)
        neg_afterReLU = nn.ReLU(graph, neg_xW1_plusb1)

        x2W2 = nn.MatrixMultiply(graph, afterReLU, self.W2)
        neg_x2W2 = nn.MatrixMultiply(graph, neg_afterReLU, self.W2)

        x2W2_plusb2 = nn.MatrixVectorAdd(graph, x2W2, self.b2)
        neg_x2W2_plusb2 = nn.MatrixVectorAdd(graph, neg_x2W2, self.b2)

        negated_term = -1*graph.get_output(neg_x2W2_plusb2)
        negated_neg = nn.Input(graph, negated_term)

        sum_terms = nn.Add(graph, x2W2_plusb2, negated_neg)
        
        # x2W2 = nn.MatrixMultiply(graph, afterReLU, self.W2)
        # x2W2_plusb2 = nn.MatrixVectorAdd(graph, x2W2, self.b2)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, sum_terms, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            output = graph.get_output(sum_terms)
            # print "Output matrix size:", output.shape
            return output


        # At each iteration, we first calculate a loss that measures how
        # good our network is. The graph keeps track of all operations used
        """
        graph = nn.Graph([self.W1, self.b1, self.W2, self.b2])

        input_x = nn.Input(graph, x)
        xW1 = nn.MatrixMultiply(graph, input_x, self.W1)
        xW1_plusb1 = nn.MatrixVectorAdd(graph, xW1, self.b1)
        afterReLU = nn.ReLU(graph, xW1_plusb1)
        x2W2 = nn.MatrixMultiply(graph, afterReLU, self.W2)
        f_pos = nn.MatrixVectorAdd(graph, x2W2, self.b2)

        # print "pos side output", graph.get_output(x2W2_plusb2)

        neg_x = np.multiply(x,np.array([-1]))
        input_x_neg = nn.Input(graph, neg_x)
        xW1_neg = nn.MatrixMultiply(graph, input_x_neg, self.W1)
        xW1_plusb1_neg = nn.MatrixVectorAdd(graph, xW1_neg, self.b1)
        afterReLU_neg = nn.ReLU(graph, xW1_plusb1_neg)
        x2W2_neg = nn.MatrixMultiply(graph, afterReLU_neg, self.W2)
        f_neg = nn.MatrixVectorAdd(graph, x2W2_neg, self.b2)

        negated_term = -1*graph.get_output(f_neg)
        negated_neg = nn.Input(graph, negated_term)

        print "neg side output", graph.get_output(negated_neg)

        sum_sides = nn.Add(graph, f_pos, negated_neg)
        

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, sum_sides, input_y)
            return graph

        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            output = graph.get_output(sum_sides)
            return output

        """

class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.5
        layerSize = 400

        self.W1 = nn.Variable(784,layerSize)
        self.b1 = nn.Variable(layerSize)
        self.W2 = nn.Variable(layerSize,10)
        self.b2 = nn.Variable(10)
        # self.W3 = nn.Variable(layerSize,10)
        # self.b3 = nn.Variable(10)


    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "*** YOUR CODE HERE ***"
        # At each iteration, we first calculate a loss that measures how
        # good our network is. The graph keeps track of all operations used
        graph = nn.Graph([self.W1, self.b1, self.W2, self.b2])
        input_x = nn.Input(graph, x)
        xW1 = nn.MatrixMultiply(graph, input_x, self.W1)
        xW1_plusb1 = nn.MatrixVectorAdd(graph, xW1, self.b1)
        afterReLU = nn.ReLU(graph, xW1_plusb1)
        x2W2 = nn.MatrixMultiply(graph, afterReLU, self.W2)
        x2W2_plusb2 = nn.MatrixVectorAdd(graph, x2W2, self.b2)
        # afterReLU2 = nn.ReLU(graph, x2W2_plusb2)
        # x3W3 = nn.MatrixMultiply(graph, afterReLU2, self.W3)
        # x3W3_plusb3 = nn.MatrixVectorAdd(graph, afterReLU2, self.b3)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SoftmaxLoss(graph, x2W2_plusb2, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            output = graph.get_output(x2W2_plusb2)
            maxval = np.max(output,1)

            for row in range(np.size(output,0)):
                max_in_row = np.max(output[row,:])
                for col in range(np.size(output,1)):
                    if output[row,col] == max_in_row:
                        output[row,col] = 1
                    else:
                        output[row,col] = 0


            # for idx in range(size(output,0)):
            #     if output[idx] == maxval:
            #         output[idx] = 1
            #     else:
            #         output[idx] = 0

            # print "Output matrix size:", output.shape
            return output


class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.03
        layerSize = 100
        # print "batchSize", batchSize
        self.W1 = nn.Variable(self.state_size,layerSize)
        self.b1 = nn.Variable(layerSize)
        self.W2 = nn.Variable(layerSize,self.num_actions)
        self.b2 = nn.Variable(self.num_actions)


    def run(self, states, Q_target=None):
        """
        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"
        graph = nn.Graph([self.W1, self.b1, self.W2, self.b2])
        input_states = nn.Input(graph, states)
        xW1 = nn.MatrixMultiply(graph, input_states, self.W1)
        xW1_plusb1 = nn.MatrixVectorAdd(graph, xW1, self.b1)
        afterReLU = nn.ReLU(graph, xW1_plusb1)
        x2W2 = nn.MatrixMultiply(graph, afterReLU, self.W2)
        x2W2_plusb2 = nn.MatrixVectorAdd(graph, x2W2, self.b2)

        if Q_target is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, Q_target)
            loss = nn.SquareLoss(graph, x2W2_plusb2, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            output = graph.get_output(x2W2_plusb2)
            # print "Output matrix size:", output.shape
            return output

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.d = 100 # sufficiently large

        self.learning_rate = 0.075
        layerSize = 50
        self.W1 = nn.Variable(5,layerSize)
        self.b1 = nn.Variable(layerSize)
        self.W2 = nn.Variable(layerSize,5)
        self.b2 = nn.Variable(5)

    def run(self, xs, y=None):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]

        "*** YOUR CODE HERE ***"
        graph = nn.Graph([self.W1, self.b1, self.W2, self.b2])
        input_x = nn.Input(graph, np.array(xs))

        h, xW1, xW1_plusb1, xW1_plusb1c, afterReLU, x2W2 = [], [], [], [], [], []
        h.append(nn.Input(graph, np.zeros_like(y)))

        for i in range(1,self.d):
            c = xs[i]
            xW1.append(nn.MatrixMultiply(graph, h[i-1], self.W1))
            xW1_plusb1.append(nn.MatrixVectorAdd(graph, xW1, self.b1))
            xW1_plusb1c.append(nn.MatrixVectorAdd(graph, xW1_plusb1, c))
            afterReLU.append(nn.ReLU(graph, xW1_plusb1c))
            x2W2.append(nn.MatrixMultiply(graph, afterReLU, self.W2))
            h.append(nn.MatrixVectorAdd(graph, x2W2, self.b2))

        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SoftmaxLoss(graph, h[-1], input_y)
            return graph
        else:
            "*** YOUR CODE HERE ***"
            output = graph.get_output(h[-1])

            for row in range(np.size(output,0)):
                max_in_row = np.max(output[row,:])
                for col in range(np.size(output,1)):
                    if output[row,col] == max_in_row:
                        output[row,col] = 1
                    else:
                        output[row,col] = 0
            return output
