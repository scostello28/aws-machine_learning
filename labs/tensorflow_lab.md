## TensorFlow Lab

**Introduction**

SSH tunnels allow you to connect to ports on a remote server through the encrypted SSH channel. This allows you to securely connect to ports on the remote server that you otherwise wouldn't be able to because of system firewall, or security group rules. In this Lab Step, you will establish an SSH connection with a tunnel from port 8000 on your local system to port 8888 on the remote server. The tunnel will allow you to connect to a Jupyter Notebook server later in the Lab to interactively develop Tensorflow machine learning models after you learn the basics at the command-line.

Different instructions are provided for Linux/macOS, and Windows.



**Instructions**
1. Navigate to the EC2 Management Console and copy the **IPv4 Public IP** address of the Lab instance.

*Note:* It may take a minute or two for the instance to appear in the list. Refresh the list every 15 seconds until it appears.

2. Proceed to the **Connecting using Linux/macOS** or **Connecting using Windows** instructions depending on your local operating system.


**Connecting using Linux/macOS**

Linux distributions and macOS include an SSH client that accepts standard PEM keys. Complete the following steps to connect using the included terminal applications:

a. Open your terminal application. If you need assistance finding the terminal application, search for terminal using your operating system's application finder or search commands.

b. Enter the following command and press Enter:
`ssh -i /Path/To/Your/KeyPair.pem ubuntu@YourIPv4Address -L127.0.0.1:8000:127.0.0.1:8888`

`ssh -i /Users/sec/Documents/aws/keys/468571902526.pem ubuntu@52.27.89.153 -L127.0.0.1:8000:127.0.0.1:8888`

where the command details are:

ssh initiates the SSH connection.

-i specifies the identity file.

/Path/To/Your/Keypair.pem specifies the location and name of your key pair. An example location might be /Home/YourUserName/Downloads/KeyPair.pem.

My key location: `/Users/sec/Documents/aws/keys/468571902526.pem`

YourIPv4Address is the IPv4 address noted earlier in the instructions.

`52.27.89.153`

-L specifies to bind 127.0.0.1:8000 on your local machine to 127.0.0.1:8888 on the remote machine.

*Note:* Your SSH client may refuse to start the connection due to key permissions. If you receive a warning that the key pair file is unprotected, you must change the permissions. Enter the following command and try the connection command again:

`chmod 600 /Path/To/Your/KeyPair.pem`

`chmod 600 /Users/sec/Documents/aws/keys/468571902526.pem`

### Learning the Basics of Tensorflow

**Introduction**
TensorFlow is a popular framework used for machine learning. It works by defining a dataflow graph. *Tensors*, or arrays of arbitrary dimension, *flow* through the graph performing operations defined by the nodes in the graph. Machine learning algorithms can be modeled using this kind of dataflow graph.

When you write code using TensorFlow, there are two phases: graph definition, and evaluation. You define the entire computation graph before executing it. With this strategy, TensorFlow can scan the graph and perform optimizations on the graph to reduce computation time, increase parallelism. These two phases are something to keep in mind when developing code with TensorFlow.

In this Lab Step, you will see how to perform mathematical operations on tensors by defining small graphs in TensorFlow. These operations are the building blocks for machine learning algorithms and you will see how they can be combined to create machine learning models in future Lab Steps. TensorFlow is available for multiple programming languages, but this Lab uses Python. Python is the most common language for working with TensorFlow. You will use Python 2.7 but the Amazon Deep Learning AMI also provides support for Python 3.6.



**Instructions**
1. In your SSH shell, enter the following command to load the Python 2.7 TensorFlow virtual environment in the Deep Learning AMI:

`$ source activate tensorflow_p27`

`(tensorflow_p27) ubuntu@ip-10-0-0-244:~$`

2. Start the interactive Python interpreter by entering:

`python`

3. Enter the following import statements to import the print function and the TensorFlow module:

```python
from __future__ import print_function
import tensorflow as tf
```

The TensorFlow module is conventionally imported as **tf**.

*Note:* You will see Warning messages throughout this Lab. In general you should pay attention to the warnings. But for the short-lived Lab session, you can ignore them and still complete the Lab without any issue.



4. Define a dataflow graph with two constant tensors as input and use the `tf.add` operation to produce the output:

```python
# Explicitly create a computation graph
graph = tf.Graph()
with graph.as_default():
  # Declare one-dimensional tensors (vectors)
  input1 = tf.constant([1.0, 2.0])
  input2 = tf.constant([3.0, 4.0])
  # Add the two tensors
  output = tf.add(input1, input2)
```

The graph keeps track of all the inputs and operations you define so the results can be computed when you run a TensorFlow session with the graph.



5. Print the graph to see that it stored the inputs and operations:

```python
print(graph.get_operations())
```

```
[<tf.Operation 'Const' type=Const>, <tf.Operation 'Const_1' type=Const>, <tf.Operation 'Add' type=Add>]
```

The list of each operation corresponding to the two input constants and the addition operation are displayed. Default unique names are shown in single quotations. You can set the names when you declare the operation by passing a named argument called `name`.

6. Evaluate the graph by creating a session with the graph and calling the output.eval() function:

```python
# Evaluate the graph in a session
with tf.Session(graph = graph):
  result = output.eval()
  print("result: ", result)
```

```
result:  [4. 6.]
```

The output displays an information message letting you know that the session will run on the instance's graphics processing unit (GPU). The result is printed on the last line.


7. When you are only using a single graph in a session, you can use the default graph as shown in the following example that repeats the computation using the default graph:

```python
# Evaluate using the default graph
with tf.Session():
  input1 = tf.constant([1.0, 2.0])
  input2 = tf.constant([3.0, 4.0])
  output = tf.add(input1, input2)
  # Show the operations in the default graph
  print(tf.get_default_graph().get_operations())
  result = output.eval()
  print("result: ", result)
```

```
[<tf.Operation 'Const' type=Const>, <tf.Operation 'Const_1' type=Const>, <tf.Operation 'Add' type=Add>]
result:  [4. 6.]
```

In the above code, the default graph is implicitly passed to all Tensorflow API functions. It can be convenient to use the default graph, but you may need multiple graphs when you develop separate training and test graphs for machine learning algorithms.



8. Multiply a matrix by a vector with the following annotated example:

```python
matmul_graph = tf.Graph()
with matmul_graph.as_default():
  # Declare a 2x2 matrix and a 2x1 vector
  matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]])
  vector = tf.constant([[1.0], [2.0]])
  # Matrix multiply (matmul) the two tensors
  output = tf.matmul(matrix, vector)

with tf.Session(graph = matmul_graph):
  result = output.eval()
  print(result)
```

You have now seen how to add vectors and multiply a matrix by a vector in
TensorFlow. These two operations are the building blocks of several machine
learning algorithms. The examples have only used constant inputs so far.
TensorFlow supports using variables to allow tensors to be updated with
different values as graph evaluation proceeds.

9. Use variables to store the result of repeatedly multiplying a matrix by a
vector as in the following annotated example:

```python
# Evaluate a matrix-vector multiplication
var_graph = tf.Graph()
with var_graph.as_default():
  # Declare a constant 2x2 matrix and a variable 2x1 vector
  matrix = tf.constant([[1.0, 1.0], [1.0, 1.0]])
  vector = tf.Variable([[1.0], [1.0]])
  # Multiply the matrix and vector 4 times
  for _ in range(4):
    # Repeatedly update vector with the multiplication result
    vector = tf.matmul(matrix, vector)

with tf.Session(graph = var_graph):
  # Initialize the variables we defined above.
  tf.global_variables_initializer().run()
  result = vector.eval()
  print(result)
```

```
[[ 5.]
 [11.]]
```

The output value of *vector* shows that the value has been updated. This is
similar to what you would expect using a variable in Python. One catch is
that variables must be initialized with an explicit call.
`tf.global_variables_initializer().run()` initializes all variables in a global
variable collection. By default, every variable is added to the global variable
collection.


10. Exit the Python interpreter by entering:

```python
exit()
```

The remainder of the Lab will use Jupyter notebooks for developing with code.

11. Deactivate the TensorFlow virtual environment:
`source deactivate `

**Summary**

In this Lab Step, you performed some mathematical operations on tensors in TensorFlow. You understood how graphs are used to define computations and session are used to perform the computations defined in a graph. You also demonstrated how to use TensorFlow variables to store the results of intermediate computations.

You have only scratched the surface of TensorFlow at this point. The full list of methods in the TensorFlow module is available [here](https://www.tensorflow.org/api_docs/python/). However, what you have learned so far is enough to understand how TensorFlow can be used for machine learning.

### Starting a Jupyter Notebook Server

**Introduction**

The Python programming language is the most popular language for data science and machine learning. Jupyter notebooks are a popular way to share data science and machine learning experiments written in Python and other languages. Jupyter notebooks allow you to share documents that include code and visualizations that the users can execute and interact with from their web browser.

A Jupyter notebook server must be running in order to create and run Jupyter notebooks. The AWS Deep Learning AMI that the Lab virtual machine is built from includes the Jupyter notebook server, in addition to many commonly used packages for machine learning. In this Lab Step, you will start a Jupyter notebook server.


**Instructions**

1. In your SSH shell, enter the following command to start the Jupyter notebook server in the background:

`nohup jupyter notebook &`

The `nohup` command, stands for no hangup and allows the Jupyter notebook server to continue running even if your SSH connection is terminated. After a couple seconds a message about writing output for the process to the `nohup.out` file will be displayed:

`nohup: ignoring input and appending output to 'nohup.out'`

2. Press enter to move to a clean command prompt, and tail the notebook's log file to watch for when the notebook is ready to connect to:

`tail -f nohup.out`

The notebook is ready when you see The Jupyter Notebook is running at:...

```
...
[I 16:54:46.649 NotebookApp] The Jupyter Notebook is running at:
[I 16:54:46.649 NotebookApp] http://localhost:8888/?token=2705938e9ed46acee55543cc5fd63f5bd63805632b5c24ac
...
```

3. Press ctrl+c to stop tailing the log file.

4. Enter the following to get an authenticated URL for accessing the Jupyter notebook server:

`jupyter notebook list`

By default, Jupyter notebooks prevent access to anonymous users. After all, you can run arbitrary code through the notebook interface. The **token** URL parameter is one way to authenticate yourself when accessing the notebook server. The `/home/ubuntu` at the end of the command indicates the working directory of the server.

**Summary**

The Jupyter notebook server is now up and running.

### Creating a Neural Network in Tensorflow

**Introduction**
To provide a useful but relatively simple example of machine learning with TensorFlow, you will create a neural network with only an input and output layer that learns the relationship between two variables, x and y. The neural network will predict the value of y for a given value of x once you train the network. x and y could represent any numerical value, such as the heart rate and cholesterol level, or age and income level. For this Lab Step, the data used for training is randomly generated by perturbing values around a known linear function between x and y. The known linear function can be written as y = a*x + b where a is the slope of the line and b is the intercept. The following image depicts the function:

In the example, a=0.5 and b=2. After training the neural network, the predicted values should closely match the values of the known function.



**Neural Network Review**

This section briefly reviews what you need to know about neural networks to understand the example in this Lab Step. A neural network is made up of layers. Every neural network has input and output layers. The layers in between the input and output layer are called hidden layers. Each layer is comprised of one or more neurons. Each neuron in the hidden and output layers takes output values from the previous layer's neurons as input. A diagram of a neural network with one hidden layer with four neurons is as follows:

diagram

Each input to a neuron in the hidden and output layers is multiplied by a weight. The sum of the multiplication results are added with a bias value to produce the output value of the neuron. In linear algebra the inputs to a neuron and the neuron's weights can be represented as vectors. To multiply the weights by the inputs and add them up, you can use a dot product or inner operation. To perform all of the operations of a layer at once, you can represent each neuron's weights in a matrix and each neuron's bias in a vector.  The complete operation for a hidden layer is then multiplying the weight matrix by the input vector and adding the bias vector. This is how values flow through the neural network. You can see how similar it is to a dataflow graph in TensorFlow. The bias can be represented as an input that is always 1 and the neuron has a bias weight like it does for any other input. This reduces the operation of a layer to just multiplying a matrix by a vector. Training is a bit more complicated and has values flowing in the opposite direction. You don't need to know the details because TensorFlow wraps the complexities of training into functions that it provides.

A single neuron is able to learn linear relationships so the neural network in this example only needs one neuron. That's why the neural network doesn't need any hidden layers, which are required to learn more complex relationships in the data.



**Instructions**

1. In your SSH shell, copy the Jupyter notebook URL output by the list command, and paste it into your browser:

Recall that the tunnel is open to port *8000* on your local machine and not *8888*.

2. Replace the port *8888* with *8000* in the URL and navigate to the site:

3. Click on the **New** button above the file listing table, and select **Environment (conda_tensorflow_p27)**

4. Paste the following Python script into the cell and read through the comments and code:

```python
'''Single neuron neural network'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Draw plots inline in the notebook
%matplotlib inline

############################

# Create Data around line
# Set up sample points perturbed away from the ideal linear relationship
# y = 0.5*x + 2.5
num_examples = 60
points = np.array([np.linspace(-1, 5, num_examples),
  np.linspace(2, 5, num_examples)])
points += np.random.randn(2, num_examples)
x, y = points
# Include a 1 to use as the bias input for neurons
x_with_bias = np.array([(1., d) for d in x]).astype(np.float32)

#############################

# Training parameters
training_steps = 100
learning_rate = 0.001
losses = []

#############################

with tf.Session():
  # Set up all the tensors, variables, and operations.
  input = tf.constant(x_with_bias)
  target = tf.constant(np.transpose([y]).astype(np.float32))
  # Initialize weights with small random values
  weights = tf.Variable(tf.random_normal([2, 1], 0, 0.1))

  tf.global_variables_initializer().run()

  # Calculate the current prediction error
  y_predicted = tf.matmul(input, weights)
  y_error = tf.subtract(y_predicted, target)

  # Compute the L2 loss function of the error
  loss = tf.nn.l2_loss(y_error)

  # Train the network using an optimizer that minimizes the loss function
  update_weights = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss)

  for _ in range(training_steps):
    # Repeatedly run the operations, updating the TensorFlow variable.
    update_weights.run()
    losses.append(loss.eval())

  # Training is done, get the final values for the graphs
  w = weights.eval()
  y_predicted = y_predicted.eval()

print("Final Weights")
print("-------------")
print("Slope: {}".format(w[1][0]))
print("Intercept: {}".format(w[0][0]))

# Show the fit and the loss over time.
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.subplots_adjust(wspace=.3)
fig.set_size_inches(11, 4)
# Plot the perturbed points in blue dots
ax1.scatter(x, y, c="b", alpha=.6)
# Plot the predicted values in green dots
ax1.scatter(x, np.transpose(y_predicted)[0], c="g", alpha=0.6)

line_x_range = (-3, 7)
# Plot the predicted line in green
ax1.plot(line_x_range, [w[1] * x + w[0]
                       for x in line_x_range], "g", alpha=0.6)
# Plot the noise-free line (0.5*x + 2.5) in red
ax1.plot(line_x_range, [0.5 * x + 2.5
                        for x in line_x_range], "r", alpha=0.6)

ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Plot Loss over training steps
ax2.plot(range(0, training_steps), losses)
ax2.set_xlabel("Training step")
ax2.set_ylabel("Loss")

plt.show();
```

The code concerning TensorFlow is all inside the with `tf.Session()`: block. There are a few functions that you haven't seen before but their names accurately describe what they do. TensorFlow provides several loss functions and training algoritms. The `tf.nn.l2_loss` is a common choice, while the `tf.train.GradientDescentOptimizer` is also popular but usually not the most efficient training algorithm. It is what more advanced algorithms are based upon.

5. Click the Run button to start running the experiment.

The code will start running. It should take less than 30 seconds to see the plots that are generated after training completes. The first time is much slower than following times because libraries need to be loaded. If you run the script again it should finish in one second or less.



6. Take a moment to analyze the plots, your plots will be different due to the random nature of the code but they should resemble the following:


The predicted relationship (green) matches the noise-free relationship (red) quite closely. The plot of the loss at each step shows that the training makes slow progress after around the 50th step. In the example above, the lowest possible loss is around 40 when modeling the relationship with a straight line.



**Summary**

In this Lab Step, you created and trained a neural network and used it to predict values of a function given noisy point samples. The example was fairly simple, but a larger neural network with more layers and neurons would mainly just need to have additional matmul operations and larger dimension variables.

At this point, it's worth mentioning that TensorFlow includes several customizable models in a high-level API called Estimators. The estimators include deep neural network regressors and classifiers. Using an estimator can dramatically reduce the amount of code required but still give you a model that performs well in most cases. To learn more about estimators, check out this overview after your Lab session is complete.

### Visualizing the Learning of the Neural Network with TensorBoard

**Introduction**
TensorFlow includes a visualization tool called TensorBoard. It allows you to visualize TensorFlow graphs, histograms of variables at each step of the learning process, and more. You will annotate the neural network with operations to log summaries while the network is being trained. You will then use TensorBoard to read the logs and generate visualizations of the graph and learning process.

The summaries read by TensorBoard are created using the [tf.summary](https://www.tensorflow.org/api_docs/python/tf/summary) module. There are functions in the module to write summaries for tensors (*scalar*), *histograms*, *audio*, and *images*. You will write summaries for tensors and histograms, but the other summary types are useful when working with audio or image data. Recall that TensorFlow uses a deferred execution model by first building a graph and then evaluating the graph to obtain results. Because summary operations aren't dependent upon for computing results, they are only logging state information, the summary operations need to be explicitly evaluated in addition to evaluating output of your graph. The summary module includes a `merge_all` function to merge all summary operations into a single operation that you can conveniently evaluate instead of evaluating all individual summary operations. You will see how to evaluate both the summaries and the model training in the example code below.

**Instructions**

1. Open a new TensorFlow Python 2.7 notebook by selecting *File > New Notebook > Environment (conda_tensorflow_p27)*

2. Paste the following code that has been annotated with `tf.summary` operations to visualize logs with TensorBoard into the notebook cell:

```python
'''Single neuron neural network with TensorBoard annotations'''

from __future__ import print_function
import numpy as np
import tensorflow as tf

# Start with a clean environment each run
log_dir = '/tmp/tensorflow/nn/train' # TensorBoard logs saved here
if tf.gfile.Exists(log_dir):
  tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MakeDirs(log_dir)
tf.reset_default_graph()

def variable_summaries(name, var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('weights'):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

# Set up sample points perturbed away from the ideal linear relationship
# y = 0.5*x + 2.5
num_examples = 60
points = np.array([np.linspace(-1, 5, num_examples),
  np.linspace(2, 5, num_examples)])
points += np.random.randn(2, num_examples)
x, y = points
# Include a 1 to use as the bias input for neurons
x_with_bias = np.array([(1., d) for d in x]).astype(np.float32)

# Training parameters
training_steps = 100
learning_rate = 0.001
losses = []

with tf.Session() as sess:
  # Set up all the tensors, variables, and operations.
  input = tf.constant(x_with_bias)
  target = tf.constant(np.transpose([y]).astype(np.float32))
  # Initialize weights with small random values
  weights = tf.Variable(tf.random_normal([2, 1], 0, 0.1))

  tf.global_variables_initializer().run()

  # Calculate the current prediction error
  y_predicted = tf.matmul(input, weights)
  y_error = tf.subtract(y_predicted, target)

  # Compute the L2 loss function of the error
  loss = tf.nn.l2_loss(y_error)

  # Train the network using an optimizer that minimizes the loss function
  update_weights = tf.train.GradientDescentOptimizer(
  learning_rate).minimize(loss)

  # Add summary operations
  variable_summaries('weights', weights)
  tf.summary.histogram('y_error', y_error)
  tf.summary.scalar('loss', loss)
  merged = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

  for i in range(training_steps):
    # Repeatedly run the summary and training operations
    summary, updates = sess.run([merged, update_weights])
    summary_writer.add_summary(summary, i)

  summary_writer.close()

print('Complete')
```

The code varies from the previous Lab Step's mainly by:

Including a function called `variable_summaries` that adds summary operations for the mean, min, max, standard deviation, and histogram of a tensor's values
Including code to add summary operations and write the summaries to disk in the `log_dir` directory every iteration of the `for` loop (The code under the # Add summary operations comment)
Removing the matplotlib plotting code since TensorBoard will create the visualizations automatically from the summary operations
You use a new way to evaluate the graph with this code. The following line is used to evaluate the graph:

`summary, updates = sess.run([merged, update_weights])`

The session's run function evaluates the operations include in its argument, in this case, the merged summary and the update_weights training operations. All of the graph dependencies of the operations are also evaluated resulting in evaluating all of the graph operations.

3. Run the notebook and watch for the `Complete` message to appear.

4. In your SSH shell, re-enter the TensorFlow Python 2.7 virtual environment to have access to Tensorboard:

`source activate tensorflow_p27`

*Note*: I had to shit down the Jupyter Kernal with control-c to be able to enter this command.

5. enter the following command to start TensorBoard:

`tensorboard --logdir /tmp/tensorflow/nn`

The `--logdir` option tells TensorBoard where to find the log data saved by the TensorFlow summary FileWriter.


6. Copy the **IPv4 Public IP** address of the EC2 instance from the details section of the instance in the EC2 Console under **INSTANCES** > **Instances**.

7. Navigate to TensorBoard by opening a new browser tab and pasting the IP address followed by :6006.

`35.163.24.88:6006/`

The Cloud Academy Lab environment automatically allows incoming traffic on port 6006 in the instance's security group. Alternatively, instead of navigating to the public IP address of the instance and allowing incoming traffic on port 6006, you could have included a second SSH tunnel. Using an SSH tunnel would encrypt the traffic and be more secure than using the instance's public IP address but for the sake of the Lab, you can use an unencrypted channel.

8. Click on the **GRAPHS** tab to first see a visualization of the TensorFlow graph.

The graph visualization is split into two parts: **Main Graph**, and **Auxiliary Nodes**. Auxiliary nodes are separated out from the main graph to avoid cluttering up the graph. For this relatively simple graph it is less of a problem than for larger graphs. The legend is located in the bottom-left corner. Take a minute to see that the graph corresponds to the default graph defined in the code. This view can make it easier to understand TensorFlow graphs compared to inspecting code. The dependencies are clearly indicated and the width of each edge corresponds to the magnitude of a tensor's dimension. So wider edges are for larger tensor's.


9. Click on various nodes in the graph and double-click on **Namespace** nodes to expand them and reveal their inner details, for example for the **gradients** namespace.

The information panel in the upper-right corner tells you about your selection including inputs and outputs, dimensions, and type of operation when relevant.



10. Return to the **SCALARS** tab that was open when you first navigated to TensorBoard.



11. Examine the **loss, weights/summaries/max, weights/summaries/mean** and **weights/summaries/min** chart and confirm that they match your expectations.

The **loss** plot closely resembles the loss plot from the previous Lab Step. The max and the min plots show the max and min parameter in the weights tensor. The max corresponds to the b value/intercept in y = a*x + b and the min value corresponds to a. There are only two elements in the weights tensor in this case. You can see that the min converges around 0.5 which is the noise-free value for a, and the max converges around 2.5, which is the noise-free value for b.

Each chart shows a dark orange line and a faded orange line. The dark orange line is the smoothed representation of the data. You can configure the **Smoothing** parameter in the menu to the left of the plots. Smoothing is more useful when the data is more noisy and hard to uncover the trend without some smoothing of the data. You can mouse over the line in any plot to see the non-smoothed value of the data. You can also draw a zoom box to zoom into an area of interest in a plot.



12. Navigate to the **DISTRIBUTIONS** tab and click the box icon below the **y_error** plot to expand it.

The distributions show you the range of values in a tensor at each of the recorded iterations. Because the weights only have two values, you mainly see a single shaded region. The **y_error** histogram is made up of 60 error values. The darkest color band represents the range of values that are within one standard deviation of the mean/average value. Lighter shades represent the range of values within high multiples of standard deviation away from the mean. For example, the range of values within two standard deviations of the mean is shown by the second darkest band. As a sign that training is improving the predictions, the bands get narrower the further to the right you look in the plot.

*Notre:* I think this means 60 error values per learning step--so each increment on the x-axis is its own histogram.



13. Navigate to the **Histograms** tab and expand both plots to reveal more detail.

The histograms plots are similar to the distributions plots you inspected except instead of focusing on ranges of values centered about the mean, the frequency of values in different bins is shown. Different iterations are shown by the depth of the plot with the latest plot in the foreground. You can move your mouse over the plot to highlight different iterations. You can clearly see the two weights converging around 0.4 and 2.7 in the **weights/summaries/histogram** histogram. The **y_error** histogram illustrates the reduction in the range of errors with more training and also shows two peaks where the most frequent errors are around -1.25 and 0.5.



14. In the SSH shell, press ctrl+c to quit TensorBoard.



**Summary**

In this Lab Step, you learned how to write summaries in TensorFlow code that can be interpreted and visualized by TensorBoard. It can be more convenient to work with TensorBoard than writing custom visualization code as long as TensorBoard includes a visualization of what you want to see. It's also useful to know that TensorBoard automatically refreshes the visualizations every 30 seconds. For larger problems that can run for hours or longer, you can watch the progress of the learning in TensorBoard.

### Serving a Model with TensorFlow Serving

**Introduction**

You can serve your TensorFlow models using a system called TensorFlow Serving. Serving a model means that clients can access models to make predictions through an API. TensorFlow Serving supports serving TensorFlow models out-of-the-box and is designed for use in production environments.

To serve a model, you can use the `tensorflow_model_server` binary that is included in the Amazon Deep Learning AMI. You need to serialize your model for TensorFlow Serving to be able to serve it. TensorFlow includes the SavedModelBuilder module to simplify the process. You need to tell the builder the signature of the prediction model. The signature tells the builder the type and shape of the inputs and outputs of the model. The model is serialized and saved to disk using Google's [protocol buffer](https://developers.google.com/protocol-buffers/) serialization format and produces files with a `.pb` file extension. TensorFlow Serving supports versioning allowing you to easily serve multiple versions of a model.

The graph you have worked with until now needs to be modified to support serving. The graph had been using constant inputs and was focused on training. When you serve a model you need to use placeholders for inputs so that a client can provide whatever input values they need to make predictions for. The code in this Lab Step indicates tensors used for training by appending train_ to their name. The new graph has separate paths for training and making predictions.



**Instructions**

1. Open a new TensorFlow Python 2.7 notebook by selecting **File > New Notebook > Environment (conda_tensorflow_p27)**.



2. Paste the following code that builds and saves the trained neural network model into the Jupyter notebook cell:

```python
'''Export single neuron neural network model for TensorFlow Serving'''

from __future__ import print_function
import os
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_integer('model_version', 1, 'Model version number')
tf.app.flags.DEFINE_string('export_dir', '/tmp/nn', 'Export model directory')
FLAGS = tf.app.flags.FLAGS

# Set up sample points perturbed away from the ideal linear relationship
# y = 0.5*x + 2.5
num_examples = 60
points = np.array([np.linspace(-1, 5, num_examples),
                   np.linspace(2, 5, num_examples)])
points += np.random.randn(2, num_examples)
train_x, train_y = points
# Include a 1 to use as the bias input for neurons
train_x_with_bias = np.array([(1., d) for d in train_x]).astype(np.float32)

# Training parameters
training_steps = 100
learning_rate = 0.001

with tf.Session() as sess:
  # Set up all the tensors, variables, and operations.
  input = tf.constant(train_x_with_bias)
  target = tf.constant(np.transpose([train_y]).astype(np.float32))
  # Set up placeholder for making model predictions (separate from training)
  x = tf.placeholder('float', shape=[None, 1])
  # Initialize weights with small random values
  weights = tf.Variable(tf.random_normal([2, 1], 0, 0.1))

  tf.global_variables_initializer().run()

  # Calculate the current prediction error
  train_y_predicted = tf.matmul(input, weights)
  train_y_error = tf.subtract(train_y_predicted, target)

  # Define prediction operation
  y = tf.matmul(x, weights[1:]) + weights[0]

  # Compute the L2 loss function of the error
  loss = tf.nn.l2_loss(train_y_error)

  # Train the network using an optimizer that minimizes the loss function
  update_weights = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss)

  for _ in range(training_steps):
    # Repeatedly run the operations, updating the TensorFlow variable.
    update_weights.run()

  ## Export the Model

  # Create a SavedModelBuilder
  export_path_base = FLAGS.export_dir
  export_path = os.path.join(export_path_base, str(FLAGS.model_version))
  print('Exporting trained model to', export_path)
  builder = tf.saved_model.builder.SavedModelBuilder(export_path)

  # Build signature inputs and outputs
  tensor_info_input = tf.saved_model.utils.build_tensor_info(x)
  tensor_info_output = tf.saved_model.utils.build_tensor_info(y)

  # Create the prediction signature
  prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'input': tensor_info_input},
      outputs={'output': tensor_info_output},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

  # Provide legacy initialization op for compatibility with older version of tf
  legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

  # Build the model
  builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
      'prediction':
      prediction_signature,
    },
  legacy_init_op=legacy_init_op)

  # Save the model
  builder.save()

print('Complete')
```

The code for building and saving the model is at the bottom of the code under the `## Export the Model` comment. You can also review the changes to the graph and notice the `tf.placeholder x` that is used as an input for making predictions. The placeholder is used by the prediction operation `y = tf.matmul(x, weights[1:]) + weights[0]`. Also notice the training operations are separated out by seeing how the `train_` variables are used.



3. Run the cell to build and save the model to disk.

Output
```
Exporting trained model to /tmp/nn/1
INFO:tensorflow:No assets to save.
INFO:tensorflow:No assets to write.
INFO:tensorflow:SavedModel written to: /tmp/nn/1/saved_model.pb
Complete
```

The **No assets to save/write** information messages are nothing to worry about. Assets are any external files needed by the model. Since the model doesn't depend on any external files, there are no assets to save.



4. Return to your SSH shell and start serving the model:

`tensorflow_model_server --port=9000 --model_name=nn --model_base_path=/tmp/nn`

The trained model is now being served by TensorFlow Serving on port 9000.

**Summary**
In this Lab Step, you saw how to modify the neural network's graph to make separate operation paths for training and predicting. You also saw how to use the SavedModelBuilder module to save a serialized model to disk. Lastly, you used the tensorflow_model_server to serve the model making it accessible to clients to make predictions with the trained model.

### Consuming the Model Served by TensorFlow Serving

**Introduction**

TensorFlow Serving models are consumed by clients. Client code needs to make a connection to the server and communicate using protocol buffers. Google's [gRPC](https://grpc.io/) remote procedure call library provides the ability to connect and make calls to the server. After establishing a connection, you make a call by preparing the input for model prediction and serializing it in the request. The serialized prediction output is returned by the server.

In this Lab Step, you will use Python client code to request predictions from the model being served by TensorFlow Serving in the previous Lab Step.

**Instructions**
1. Open a new TensorFlow Python 2.7 notebook by selecting **File > New Notebook > Environment (conda_tensorflow_p27)**.

2. Paste the following code that makes a request of the model being served by TensorFlow Serving at localhost:9000.

```python
from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

hostport = 'localhost:9000'


def do_prediction(hostport):

  # Create connection
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  # Initialize a request
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'nn'
  request.model_spec.signature_name = 'prediction'

  # Use evenly-spaced points for test data
  tests = temp_data = np.array([range(-1, 6, 1)]).transpose().astype(
    np.float32)

  # Set the tests as the input for prediction
  request.inputs['input'].CopyFrom(
    tf.contrib.util.make_tensor_proto(tests, shape=tests.shape))

  # Get prediction from server
  result = stub.Predict(request, 5.0) # 5 second timeout

  # Compare to noise-free actual values
  actual = np.sum(0.5 * temp_data + 2.5, 1)

  return result, actual


prediction, actual = do_prediction(hostport)
print('Prediction is: ', prediction)
print('Noise-free value is: ', actual)
```

Read through the code and comments to understand how the client works. The `grpc` and `tensorflow_serving.apis` modules provide facilities for creating the channel and working with protocol buffers.



3. Run the cell to make the request:

Output:
```
('Prediction is: ', outputs {
  key: "output"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 7
      }
      dim {
        size: 1
      }
    }
    float_val: 2.34487509727
    float_val: 2.73685598373
    float_val: 3.12883687019
    float_val: 3.52081775665
    float_val: 3.91279888153
    float_val: 4.30478000641
    float_val: 4.69676065445
  }
}
)
('Noise-free value is: ', array([2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ], dtype=float32))
```

The `float_vals` in the prediction output protocol buffer are the predicted values for each of the inputs you requested. Your values will differ from the image above, but you should see that they are close to the noise-free values.



**Summary**

In this Lab Step, you created a TensorFlow Serving client that makes predictions using the model served in the previous Lab Step.

You have now seen a comprehensive view of working with TensorFlow using the Amazon Deep Learning AMI. You developed a model in TensorFlow, analyzed the learning process in TensorBoard, served the trained model with TensorFlow Serving, and consumed the model using Python client code. You would follow the same steps for adding your own model into production.
