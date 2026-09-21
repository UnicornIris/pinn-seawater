# Draft: Section 2 — first- and second-order gradients

Drafted text for Section 2.2-2.3 of the paper (LaTeX-ready math). It matches what
`qcpinn_timing_profile.py` measures (`forward_only`, `first_grad`, `second_grad`) and how
`PINN._pde_residual` / `_bc_residual` in `qcpinn_seawater.py` use the derivatives.

---

## 2.2 Derivatives in a physics-informed loss

Let $\hat T_\theta(z,t)$ denote the network output with trainable parameters $\theta$, and let
$x = (z,t)$ be its input. Training a PINN requires derivatives of two different kinds, and it
is important to keep them apart.

**Derivatives with respect to the input.** The governing equation is enforced through the
residual
$$
r_\theta(z,t) \;=\; \frac{\partial \hat T_\theta}{\partial t} \;-\; \kappa\,\frac{\partial^2 \hat T_\theta}{\partial z^2},
$$
where $\kappa$ is the thermal diffusivity. Evaluating $r_\theta$ therefore requires

- the **first-order gradient** of the network output with respect to its input,
  $\nabla_x \hat T_\theta = \big(\partial \hat T_\theta/\partial z,\; \partial \hat T_\theta/\partial t\big)$,
  which supplies the time derivative in the residual and the boundary residuals of the
  Neumann ($\partial \hat T_\theta/\partial z = 0$) and Robin
  ($\partial \hat T_\theta/\partial z + \hat T_\theta = 0$) problems; and
- the **second-order gradient** $\partial^2 \hat T_\theta/\partial z^2$, obtained by
  differentiating $\partial \hat T_\theta/\partial z$ once more with respect to $z$, which
  supplies the diffusion term.

We compute both by reverse-mode automatic differentiation, retaining the computation graph
after the first pass so that it can be differentiated a second time. The second-order
gradient therefore costs a second backward pass through the graph created by the first
one, on top of the forward pass and the first-order pass.

**Derivatives with respect to the parameters.** The optimizer additionally needs
$\partial \mathcal L/\partial\theta$, where $\mathcal L$ is the loss built from $r_\theta$, the
boundary residuals, and (for inverse problems) the data misfit. Because $r_\theta$ itself
contains input derivatives, this is a further backward pass through the graph of the
first- and second-order gradients above. We refer to $\partial \hat T_\theta/\partial x$ and
$\partial^2 \hat T_\theta/\partial z^2$ as first- and second-order *input* gradients, and to
$\partial \mathcal L/\partial\theta$ as the *parameter* gradient.

## 2.3 What we time

We time three cumulative stages, each on the same batch of collocation points:

1. **forward**: evaluate $\hat T_\theta(x)$;
2. **first-order gradient**: the forward pass plus $\nabla_x \hat T_\theta$;
3. **second-order gradient**: the forward pass, $\nabla_x \hat T_\theta$, and
   $\partial^2 \hat T_\theta/\partial z^2$.

The stages are cumulative, so the *second-order-to-forward ratio* we report is the cost of
computing everything needed for the PDE residual relative to a plain evaluation of the
network. The parameter-gradient pass used by the optimizer is not part of these timings,
so the cost of a full training step is higher than the second-order-gradient time by at
least that additional backward pass.

## 2.4 Why higher-order gradients matter for hybrid quantum models

For a classical network, each additional differentiation adds a backward pass whose cost is
of the same order as the forward pass. For a hybrid network the quantum layer is part of the
graph being differentiated. On a classical simulator using backpropagation, each backward
pass traverses the operations that simulate the circuit, so the cost of every stage grows
with the simulated circuit (qubit count and depth). On quantum hardware, gradients of
circuit outputs are usually obtained with the parameter-shift rule, which evaluates the
circuit at shifted parameter values; higher-order derivatives require the rule to be applied
repeatedly. We measure the simulator case only (Section 4); we do not claim that the
simulator ratios carry over to hardware.

---

## Notes for revision

- Symbols: the code calls the diffusivity `kz`; replace $\kappa$ with the paper's existing
  symbol if it uses another one.
- The inverse-problem loss also contains a data-misfit term; it is mentioned in one clause
  above and does not affect the timed stages.
- The claim about the parameter-shift rule in 2.4 is background, not a result of this work.
  Add a citation or soften it if a reviewer might challenge it.
