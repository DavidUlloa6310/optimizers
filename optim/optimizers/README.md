# Optimizers Explained

## Stochastic Gradient Descent

Stochastic Gradient Descent is the simplest optimizer you could implement. In the ["Intro to Gradient Descent"](https://gist.github.com/DavidUlloa6310/b8e93cb20e60f2a9d626a9bfe0a8b0bf) notebook, this is the optimizer we implement, because its simplicity.

$$
w_{t+1} = w_t - \eta g(w_t)
$$

## SGD w/ Momentum

The simplest addition to SGD would be considering the "momentum" of your parameter updates. Given the following equation:

$$
\begin{align*}
v_{t+1} = \rho v_t - \eta g(w_t) \\
w_{t+1} = w_{t} + v_{t+1}
\end{align*}
$$

$v_t$ can be seen as the velocity carried by our gradient update, making $w_t$ the position. Important to know, $\rho$, is unique for each parameter, so you'll need to create a PyTree of velocities.

## SGD w/ Nesterov Momentum

One thing you might have not considered is, in our SGD w/ momentum, we calculate velocity, $v_{t+1}$ with respect to the previous $w_t$. Doing this means our newly weight matrix does not consider the direction in which our velocity is pushing it in.

To fix this, we can add this velocity to our weights before getting its gradients.

$$
\begin{align*}
v_{t+1} = \rho v_t - \eta g(w_t + \rho v_t) \\
w_{t+1} = w_t + v_{t+1}
\end{align*}
$$

Computing $g(w_t + \rho v_t)$ needs a second forward pass, so in practice (and in `nesterov_sgd.py`) we use the equivalent reformulation from Sutskever et al., which only needs $g(w_t)$:

$$
\begin{align*}
v_{t+1} = \rho v_t + g(w_t) \\
w_{t+1} = w_t - \eta (g(w_t) + \rho v_{t+1})
\end{align*}
$$

The key detail is that the step uses the new velocity $v_{t+1}$. Using $v_t$ instead gives $w_{t+1} = w_t - \eta v_{t+1}$, which is plain momentum.

## AdaGrad

AdaGrad takes an interesting approach by keep a per-parameter sum of the gradients and updates each accordingly. You might notice the $\sqrt{v_{t_1}}$ is oddly similar to an L2 norm.

The funny part - it is. We end up manipulating the "learning rate" (or multiplier to this gradient) based on this L2 norm, such that parameters which a higher gradient L2 norm update less, and those with a lower gradient L2 norm update more.

$$
\begin{align*}
v_{t+1} = v_{t} + g(w_i)^2 \\
w_{t+1} = w_{t} -\frac{\eta}{ϵ + \sqrt{v_{t+1}}}g(w_{t}) \\
\end{align*}
$$

where $\epsilon$ is a very small value (on the order of $10^{-9}$) to prevent a division by 0.

## RMSProp

Very similar to AdaGrad, RMSProp introduces a discount factor, $\beta$, which controls the influence of new gradients vs old ones in our velocity term.

$$
\begin{align*}
v_{t+1} = \beta v_{t} + (1 - \beta) * g(w_t)^2 \\
w_{t+1} = w_{t} - \frac{\eta}{\epsilon + \sqrt{v_{t+1}}} g(w_t)
\end{align*}
$$

## Adam

Finally, we get to Adam - the go-to optimizer for most ML workloads. Adam is very similar to RMSProp, but re-introduces the momentum term used in previous optimizers. Momentum helps prevents our model from getting stuck in local minima, while our velocity allows our gradients to jump quickly through "flat" areas of our loss.

$$
\begin{align*}
m_{t+1} = \beta_1 m_t + (1 - \beta_1) * g(w_t) \\
v_{t+1} = \beta_2 v_t + (1 - \beta_2) * g(w_t)^2 \\
\hat{m_{t+1}} = \frac{m_{t+1}}{1-\beta_1^{t+1}} \\
\hat{v_{t+1}} = \frac{v_{t+1}}{1-\beta_2^{t+1}} \\
w_{t+1} = \frac{\eta}{\epsilon + \sqrt{\hat{v_{t+1}}}} * \hat{m_{t+1}}
\end{align*}
$$

As you might notice, we also have $B_1^{t+1}$ anad $B_2^{t+1}$. These are meant to scale $m_{t+1}$ and $v_{t+1}$, where the more time steps pass, the smaller each of these beta values become. Initially, $m_{t+1}$ and $v_{t+1}$ are larger because of this and become slower over time.

## Shampoo

Every optimizer above (other than Muon) adapts each parameter on its own, which is the same as using a *diagonal* preconditioner. Full-matrix AdaGrad would instead keep the whole $n \times n$ matrix of gradient outer products, which is impossible for any real layer. Shampoo ([Gupta, Koren & Singer, 2018](https://arxiv.org/abs/1802.09568)) sits in between: for a weight matrix $W \in \mathbb{R}^{m \times n}$ it keeps one preconditioner per dimension.

$$
\begin{align*}
L_{t+1} = L_t + G_t G_t^T \\
R_{t+1} = R_t + G_t^T G_t \\
w_{t+1} = w_t - \eta L_{t+1}^{-1/4} G_t R_{t+1}^{-1/4}
\end{align*}
$$

$L$ is $m \times m$ and $R$ is $n \times n$, so the memory cost is $m^2 + n^2$ instead of $(mn)^2$, while still capturing correlations between rows and between columns. $L_0$ and $R_0$ start at $\epsilon I$.

For a tensor of order $k$ (like a $5 \times 5 \times C_{in} \times C_{out}$ conv kernel) the same idea generalizes: each mode $i$ keeps $H_i \mathrel{+}= G_{(i)} G_{(i)}^T$, where $G_{(i)}$ flattens every other dimension, and the gradient is multiplied by $H_i^{-1/2k}$ along every mode. A bias vector ($k = 1$) is then just full-matrix AdaGrad.

Two practical tweaks in our implementation:

* The inverse roots need an eigendecomposition, which is by far the slowest part, so they are only recomputed every `update_every` steps (10 by default).
* A dimension larger than `max_precond_dim` (2048) uses a diagonal statistic instead of a full matrix. Otherwise the $10{,}000 \times 1$ IMDB layer would need a $10{,}000 \times 10{,}000$ preconditioner.
