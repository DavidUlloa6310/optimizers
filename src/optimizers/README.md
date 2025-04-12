# Optimizers Explained

## Stochastic Gradient Descent

Stochastic Gradient Descent is the simplest optimizer you could implement. In the ["Intro to Gradient Descent"](https://gist.github.com/DavidUlloa6310/b8e93cb20e60f2a9d626a9bfe0a8b0bf) notebook, this is the optimizer we implement, because its simplicity.

$$
w_{t+1} = w_t - \eta\,g(w_t)
$$

## SGD w/ Momentum

The simplest addition to SGD would be considering the "momentum" of your parameter updates. Given the following equation:

$$
w_{t+1} = w_{t} + v_{t+1} \\\\
v_{t+1} = \rho v_t - \eta g(w_t)
$$

$v_t$ can be seen as the velocity carried by our gradient update, making $w_t$ the position. Important to know, $\rho$, is unique for each parameter, so you'll need to create a PyTree of velocities.

## SGD w/ Nesterov Momentum

One thing you might have not considered is, in our SGD w/ momentum, we calculate velocity, $v_{t+1}$ with respect to the previous $w_t$. Doing this means our newly weight matrix does not consider the direction in which our velocity is pushing it in.

To fix this, we can add this velocity to our weights before getting its gradients.

$$

v_{t+1} = \rho v_t - \eta g(w_t + \rho v_t) \\
w_{t+1} = w_t + v_{t+1}

$$

## AdaGrad

AdaGrad takes an interesting approach by keep a per-parameter sum of the gradients and updates each accordingly. You might notice the $\sqrt{v_{t_1}}$ is oddly similar to an l2 norm.

The funny part - it is. We end up manipulating the "learning rate" (or multiplier to this gradient) based on this l2 norm, such that parameters which a higher gradient l2 norm update less, and those with a higher gradient l2 norm update more.

$$
v_{t+1} = v_{t} + g(w_i)^2 \\\\
w_{t+1} = w_{t} -\frac{\eta}{ϵ + \sqrt{v_{t+1}}}g(w_{t})
$$

where $\epsilon$ is a very small value (on the order of $10^{-9}$) to prevent a division by 0.

## RMSProp

Very similar to AdaGrad, RMSProp introduces a discount factor, $\beta$, which controls the influence of new gradients vs old ones in our velocity term.

$$
v_{t+1} = \beta v_{t} + (1 - \beta) * g(w_t)^2 \\\\
w_{t+1} = w_{t} - \frac{\eta}{\epsilon + \sqrt{v_{t+1}}} g(w_t)
$$

## Adam

Finally, we get to Adam - the go-to optimizer for most ML workloads. Adam is very similar to RMSProp, but re-introduces the momentum term used in previous optimizers. Momentum helps prevents our model from getting stuck in local minima, while our velocity allows our gradients to jump quickly through "flat" areas of our loss.

$$
m_{t+1} = \beta_1 m_t + (1 - \beta_1) * g(w_t) \\\\
v_{t+1} = \beta_2 v_t + (1 - \beta_2) * g(w_t)^2 \\\\
\hat{m_{t+1}} = \frac{m_{t+1}}{1-\beta_1^{t+1}} \\\\
\hat{v_{t+1}} = \frac{v_{t+1}}{1-\beta_2^{t+1}} \\\\
w_{t+1} = \frac{\eta}{\epsilon + \sqrt{\hat{v_{t+1}}}} * \hat{m_{t+1}}
$$

As you might notice, we also have $B_1^{t+1}$ anad $B_2^{t+1}$. These are meant to scale $m_{t+1}$ and $v_{t+1}$, where the more time steps pass, the smaller each of these beta values become. Initially, $m_{t+1}$ and $v_{t+1}$ are larger because of this and become slower over time.
