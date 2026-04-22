# State Space & Configuration

## Overview
At the most fundamental level, a system within FDS has a **State**. This state defines the exact, current configuration of the system at any given tick. 

The **State Space** is the mathematical set of all possible states that the system could theoretically occupy. FDS is designed to be coordinate-agnostic, meaning the state space can be as simple or as dimensionally complex as the simulation requires.

## Examples of State Spaces

The definition of a state depends entirely on what you are simulating. 

### Example 1: The 1D Random Walker
If you are simulating a particle moving along a simple line, the state space is a 1-dimensional discrete grid.
* **A Single State:** `[5]` (The particle is at position 5)
* **The State Space:** The set of all integers `[-∞, ..., -1, 0, 1, ..., ∞]`, bounded by your topology.

### Example 2: The 2D Quantum Box (Multi-Entity)
If you are simulating multiple particles interacting inside a 2D potential well, the state configuration becomes an ensemble of coordinates.
* **A Single State:** `[[x1, y1], [x2, y2], [x3, y3]]`
* **The State Space:** The set of all valid `(x, y)` coordinate permutations within the defined boundaries of the box.

## Domain vs. Kernel Representation
Because FDS utilizes a Dual-Flow Architecture, the concept of a "State" exists in two forms:

1. **The Domain Layer (Human-Friendly):** Users define states using the rich Python `State` class (e.g., `State(np.array([10.0, -5.0]))`). This allows for easy manipulation, caching, and topology checks.
2. **The Kernel Layer (Hardware-Friendly):** Before entering the hot-loop, the Component Manager flattens these `State` objects into contiguous C-arrays (e.g., a flat `np.float64` array). The execution kernel only sees raw numbers, ensuring maximum performance without object-oriented overhead.