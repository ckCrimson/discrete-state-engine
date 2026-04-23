# The Generator: Wave Expansion & Path Generation

## Overview
In the FDS engine, the **Generator** is the computational workhorse of the simulation. If **Topology** provides the map and **Field Algebra** provides the mathematical rules, the Generator is the engine that drives the system forward. 

During the execution hot-loop, the Generator explores the state space, calculates transition weights, and expands the multi-step frontiers. It is the architectural embodiment of "Wave Expansion."

## The Generation Pipeline
When the Component Manager signals the Generator to step forward (`runner.next(apply_generator=True)`), it executes a highly optimized sequence bypassing the Python GIL:

1. **State Retrieval:** It reads the current active states ($S_t$) and their associated field values ($\psi_t$).
2. **Path Search:** It queries the Topology to find the Reachable Set (the neighbors, $V[s]$) for every active state.
3. **Algebraic Transition:** It applies the Field Algebra's transition function to the movement across the topological edges.
4. **Superposition:** It accumulates the newly transitioned field values at the destination states. If multiple paths lead to the exact same state, the Generator uses the Algebra's addition rule ($\oplus$) to superimpose the waves.

Mathematically, for a single step, the new field value at state $x$ is the sum of all fields from previous states $y$ transitioning into it, weighted by the transition rule $w$:
$$\psi_{t+1}(x) = \sum_{y \in R[x]} \psi_t(y) \otimes w(y \to x)$$

## Example: The 2D Quantum Particle
In a quantum system, a particle does not hold a single deterministic position; it exists as a complex probability wave. The Generator handles this expansion before the wave collapses.

Using our 2D Infinite Potential Well as an example:
* **The State:** A particle at coordinate `(0, 0)`.
* **The Field:** A complex amplitude of `1.0 + 0.0j`.
* **The Transition Rule:** `phase_shift_transition`, which calculates the spatial angle $\theta$ and applies the complex rotation $e^{i\theta} = \cos(\theta) + i \sin(\theta)$.

When the Generator runs for $l=5$ steps (`steps=5`), it recursively maps the topological neighbors. Instead of moving the particle, it *smears* the complex amplitude across the $F^5[s]$ frontier. When multiple paths cross the same node, their complex phases either constructively add together or destructively cancel out exactly to zero.

## Sparse Data Structures (CSR)
Because state spaces grow exponentially with every step, the Generator cannot allocate memory for an entire infinite grid. It only tracks the states that actually exist in the current basin or frontier.

To achieve this at hardware speeds, the FDS Generator utilizes **Compressed Sparse Row (CSR)** matrices (e.g., `NumbaComplexCSRGeneratorStorage`). 
* The CSR format perfectly maps the relationships between `initial_states`, `topological_edges`, and `resulting_states`.
* It allows the Execution Kernel to perform vectorized matrix multiplications for path expansion, evaluating millions of potential quantum futures in milliseconds.

---

## Visualizing the Wave Expansion

In our 2D Quantum Particle test (`run_complex_box_test`), the Generator expands the probability wave outward across the grid before any observation is made. 

Below is the telemetry from the Generator mapped to a 2D grid. As the multi-step frontier ($F^l[s]$) pushes outward from the origin, the Field Algebra calculates the complex phase interactions, resulting in the visible interference fringes (ripples) before the wave eventually collapses.

> **Quantum Wave Propagation & Interference:**
>
> ![Complex Wave Propagation](particle_grid_simulator/test/generator/plot/complex_field_generation.gif)
> *The absolute square ($|\psi|^2$) of the complex probability wave as computed by the FDS Generator across multiple ticks.*