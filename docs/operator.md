# The Operator: Observation & Wave Collapse

## Overview
In the FDS Dual-Flow Architecture, the execution hot-loop is divided into two distinct phases. Phase 1 is the **Generator**, which expands the multi-step frontiers and superimposes probability fields. Phase 2 is the **Operator**, which acts as the "Observer."

The Operator evaluates the expanded probability wave, applies systemic constraints, and resolves the system back into discrete, classical states before the next macro-tick begins. It is the architectural embodiment of Wave Collapse.

## The Resolution Mechanism
When the Component Manager signals the Operator to step (`runner.next(apply_generator=False)`), it executes a strict filtering and selection pipeline:

1. **Exclusion & Filtering:** The Operator checks the expanded frontier against physical rules. For example, it can enforce Pauli-like exclusion, ensuring no two entities claim the exact same coordinate. Invalid states are assigned a probability of `0.0`.
2. **Weight Evaluation:** It reads the Field Algebra values at the valid target states and converts them into observable weights (e.g., taking the squared norm of a complex amplitude).
3. **Re-Normalization:** Because exclusion may have eliminated some paths, the Operator re-sums the remaining valid weights to establish a new probability distribution where the total area equals `1.0`.
4. **The Pick (Collapse):** Using either a deterministic rule (e.g., "pick the highest value") or a stochastic roll (e.g., a Monte Carlo selection against a cumulative distribution), the Operator selects a single destination state from the frontier.
5. **State Update:** The Entity's vector is updated to this new classical coordinate, and its field is reset (e.g., back to a pure `1.0 + 0.0j` amplitude) ready for the next wave expansion.

## Example: The 4D Momentum Random Walker
To illustrate the Operator's coordinate-agnostic design, consider a particle tracked in a 4D phase space: $[x, y, p_x, p_y]$ (Position and Momentum).

1. **Wave Expansion:** During Phase 1, the Generator runs for $l=20$ steps. It doesn't just spread the particle across $(x, y)$ space; it smears the probability wave across all valid momenta as well. The system is in a massive state of superposition.
2. **The Born Rule Evaluation:** The Operator activates. It iterates through the 4D multi-step frontier ($F^{20}[s]$), evaluating the complex field amplitude at every $[x, y, p_x, p_y]$ coordinate to calculate the observable probability:
    $$P = |\psi|^2 = \psi_{real}^2 + \psi_{imag}^2$$
3. **Stochastic Collapse:** The Operator rolls a random float and steps through the normalized probabilities. It selects a single 4D coordinate. 
4. **Reset:** The wave vanishes. The particle now holds a definitive position and momentum, and the field amplitude at this new exact state is reset to $1.0$. 

In the very next tick, the wave begins expanding again from this newly collapsed state.

## Mapping to the Architecture
Because wave collapse requires evaluating global constraints (like checking a `master_states_ptr` for collisions across all entities), the Operator executes at the highest level of the kernel array.

* **Domain Layer:** The user defines the resolution logic (e.g., an `ExclusiveBornOperator`).
* **Component Manager:** The `OperatorComponentManager` binds this logic and injects the Fast References for the Entity States and the Generator's CSR matrices.
* **Kernel Layer:** Running completely outside the Python GIL, the JIT-compiled Operator iterates over the entire batch of entities, resolving their wave collapses sequentially or in parallel at hardware speeds.

---

## Visualizing the Collapse

Below is the telemetry from a 4D Momentum Random Walker mapped down to a 2D visual representation. 

Notice the breathing pattern: The Generator expands the wave for 20 continuous steps (the probability density spreading outward), followed by the Operator instantly collapsing it to a single point. The GIF captures 3 full iterations of this expand-and-collapse cycle.

> 🖼️ **Expansion and Collapse Iterations:**
> 
> ![4D Wave Collapse](particle_grid_simulator/test/operator/plots/4d_momentum_collapse.gif)
> *A particle undergoing 3 macro-iterations. Each cycle consists of a 20-step wave expansion (Generator) followed by an instantaneous stochastic collapse (Operator).*