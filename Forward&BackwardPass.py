#EXP3:
import tensorflow as tf

x = tf.constant([[0.0,0.0],
                 [0.0,1.0],
                 [1.0,0.0],
                 [1.0,1.0]],dtype=tf.float32) #input: 4 samples 2 features

y = tf.constant([[0.0],[1.0],[1.0],[0.0]],dtype=tf.float32) #target

w1 = tf.Variable(tf.random.normal([2,4]))
b1 = tf.Variable(tf.zeros([4]))
w2 = tf.Variable(tf.random.normal([4,1]))
b2 = tf.Variable(tf.zeros([1]))

epochs = 6000
for step in range(epochs):
    with tf.GradientTape() as tape:
        #forward pass
        h = tf.nn.relu(tf.matmul(x,w1)+b1)
        out = tf.nn.sigmoid(tf.matmul(h,w2)+b2)

        loss = tf.reduce_mean((y-out)**2)

    #backward
    grads = tape.gradient(loss, [w1,b1,w2,b2])
    for var, grad in zip([w1,b1,w2,b2],grads):
        var.assign_sub(1.0*grad) # gradient descent

    if step%200 ==0:
        print(f"Step {step}, Loss: {loss.numpy():.4f}")


print("\nPredictions:")
print(out.numpy())