# Field Algebra: Inner Product Spaces

## Overview
In the FDS engine, **Topology** defines the pathways a system can take. **Field Algebra** defines the dynamic quantities that travel along those pathways. 

Mathematically, a Field Algebra in FDS is defined as an **Inner Product Vector Space**. This strict formalization guarantees that no matter what kind of system you are simulating—from classical heat diffusion to n-dimensional quantum spin states—the engine can systematically calculate superpositions and transition weights.

## The Vector Space Fundamentals
To qualify as a valid Field Algebra, the space must define standard vector operations and their respective identities:

1. **Addition ($\oplus$):** Defines the superposition of fields when multiple paths converge on the exact same state.
2. **Additive Identity ($0$):** Represents an empty space or absolute zero probability.
    $$\psi \oplus 0 = \psi$$
3. **Multiplication ($\otimes$):** Defines how a field scales, decays, or shifts as it traverses a topological edge.
4. **Multiplicative Identity ($1$):** Represents a lossless transition with no phase shift or scaling.
    $$\psi \otimes 1 = \psi$$

## The Inner Product and The Norm
The defining feature of the Field Algebra is the inner product $\langle \psi_1, \psi_2 \rangle$. The inner product allows the engine to project vectors onto one another and, most importantly, defines the **Norm** (length) of the field vector.

In FDS, the norm is explicitly used to calculate the **weight** (or observable probability) of a state:
$$||\psi|| = \sqrt{\langle \psi, \psi \rangle}$$

* **In Classical Diffusion (Real Space $\mathbb{R}$):** The field value is a scalar probability, and the norm is simply the absolute value.
* **In Quantum Mechanics (Complex Space $\mathbb{C}$):** The field value is a probability amplitude. The weight (observable probability) is defined by the squared norm, evaluated via the complex conjugate: 
    $$||\psi||^2 = \psi \psi^* = (a + bi)(a - bi) = a^2 + b^2$$

## Dimensionality and Information Density
The dimensionality of the vector space directly correlates to the amount of information the system holds regarding transition probabilities.

* **Low Dimensionality (e.g., 1D Real):** Holds basic amplitude. Sufficient for classical random walkers or basic mass transfer. Transitions are simple scalar multiplications.
* **Medium Dimensionality (e.g., 1D Complex):** Holds amplitude *and* phase. Necessary for systems with interference patterns, where transitions require phase shifts (rotations in the complex plane).
* **High Dimensionality (e.g., n-D Tensors):** The higher the dimensionality, the richer the transition data. An n-dimensional Field Algebra can encode multi-channel probabilities, quantum spin states, or color charges, allowing the engine to evaluate highly complex transition matrices in a single execution tick.

## Execution via the Component Manager
Because the algebra is mathematically generalized, the Domain Layer simply passes the dimension count and data type (e.g., `dimensions=2, dtype=np.float64`) to the `KernelDataContract`. The Component Manager translates this into optimally packed, contiguous memory arrays, allowing the Execution Kernel to perform vectorized inner products across millions of states simultaneously.