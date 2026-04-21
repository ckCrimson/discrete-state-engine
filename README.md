# FDS: Field Dynamic System Engine

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

**Classical and Quantum stochastic modeling through a unified, hardware-accelerated pipeline.**

FDS is a modular engine designed to simulate multi-entity dynamic systems where state transitions are governed by propagating fields. To overcome the computational limits of standard object-oriented modeling, FDS implements a strict **Dual-Flow Architecture**. 

Systems are designed in a rich, human-friendly **Domain layer**, then translated via strict contracts into **Data-Oriented Kernels** for blistering, hardware-friendly execution. By bypassing the Python GIL and structuring memory into contiguous C-arrays, FDS achieves sub-second batch processing for complex path integrals and dense branching.

---

## Visual Proof: One Pipeline, Infinite Domains

Because FDS separates the physics (Domain) from the execution (Kernel), you can simulate fundamentally different mathematical realities simply by swapping the Field Algebra and Operator contracts.



**Classical Deterministic System** | **Quantum Probabilistic System**
--- | ---
![Classical Simulation](particle_grid_simulator/test/dynamic_system/plots/fast_bouncing_box.gif) | ![Quantum Simulation](particle_grid_simulator/test/dynamic_system/plots/quantum_propagation_collapse.gif)
*Entities reacting to hard topological boundaries and classical gradients.* | *Entities experiencing phase-shift interference and Born Rule wave collapse.*

> **Note:** Both simulations above are executed through the exact same FDS pipeline at C-level speeds.

## The Field Dynamic System (FDS)

At its core, FDS is a mathematical framework for modeling discrete, multi-entity stochastic systems. FDS stores the initial state of the system and orchestrates a continuous cycle of **path exploration** (wave expansion) and **state transition** (particle collapse).

> 📖 **Deep Dive:** For the exact mathematical formulation of the inner product spaces and Markov generators, read the [FDS Theoretical Formulation](docs/theory.md).

The engine operates on five theoretical pillars:

### 1. State Space (The Configuration)
The fundamental set of all possible discrete states the system can occupy. Depending on the simulation, this can be a simple 2D coordinate grid or a high-dimensional ensemble configuration space.
*🔗 See: [Defining State Spaces & Configurations](docs/state_space.md)*

### 2. Topology (The Connectivity)
If the State Space is the map, Topology is the road network. It defines the connectivity rules, determining exactly which state transitions are possible from any initial state. It is responsible for searching the valid frontier of paths the system can reach over a given number of iterations.
*🔗 See: [Topology & Boundary Mathematics](docs/topology.md)*

### 3. Fields (The Algebraic Weights)
Fields encode the mathematical rules governing transition probabilities. Formally defined as an inner product vector space (with defined addition, multiplication, unity, and null values), the Field maps a specific mathematical weight to every state. This **Field Algebra** dictates the physical reality of the simulation, whether using standard floats for classical diffusion or `complex128` arrays for quantum interference.
*🔗 See: [Implementing Custom Field Algebras](docs/algebras.md)*

### 4. The Generator (The Path Integral Explorer)
Given the Topology and Field Algebra, the Generator acts as a highly advanced, generalized Markov chain. It searches through all possible paths the system could take to reach a frontier state, accumulating the field weights for every transition along the way. This effectively performs a discrete path integral, calculating the superimposed probability field without actually moving the entities.
*🔗 See: [Generator Wave Expansion Mechanics](docs/generator.md)*

### 5. The Operator (The Observer & Collapser)
While the Generator expands the wave of possibility, the Operator collapses it into reality. It evaluates the superimposed fields, applies domain-specific rules (e.g., the Born Rule or collision avoidance), and forces the system into a single, definitive new state. 
*🔗 See: [Operator Contracts & Wave Collapse](docs/operator.md)*

---

### 🔄 The FDS Execution Flow
FDS is the culmination of these components, binding them into a strict, memory-safe execution loop:

1. **Initialize:** FDS stores the initial state and starting field.
2. **Propagate (Generator):** Expands the possibility frontier over $N$ steps, accumulating field weights.
3. **Observe (Operator):** Evaluates the generated fields and picks the final state.
4. **Collapse:** The entity adopts the new state, the field collapses to unity at that coordinate, and the cycle repeats.

## 🏗️ The Dual-Flow Architecture (Data-Oriented Design)

Most Python physics engines force a compromise: they are either easy to write but slow to execute, or fast to execute but a nightmare to maintain. FDS solves this by abandoning object-oriented processing in the simulation hot-loop, utilizing a strict **Dual-Flow Architecture** built on Data-Oriented Design (DOD) principles.

The engine separates the conceptual rules of the simulation from the hardware execution.

### The Data Flow & Dependency Injection

```mermaid
flowchart TD
    subgraph Domain ["1. Domain Layer (Human-Friendly)"]
        Logic["User Domain Logic"]
        Bridge["Inter-CM Bridge Data"]
    end

    subgraph CM ["2. Translation Layer (Component Manager)"]
        API["Public API"]
        Buffer["Command Buffer"]
        Translator["Translator / Sync Engine"]

        subgraph CS ["Component Storage"]
            Mapped[("Mapped Domain Data")]
            Optimized[("Kernel-Optimized Memory")]
        end
    end

    subgraph Interface ["Dependency Injection Boundary"]
        Contract{{"Kernel Data Contract"}}
    end

    subgraph Kernel ["3. Kernel Layer (Execution Engine)"]
        Default["Default JIT Kernel"]
        Custom["Custom Engine (C++ / CUDA / Rust)"]
    end

    %% Flow
    Logic -->|Full Bake| Translator
    Bridge --> API
    Logic --> API
    API --> Buffer
    Buffer -->|bake_incremental| Translator

    Translator <-->|Syncs| Mapped
    Translator -->|Transforms to| Optimized

    Optimized <==>|Fast Ref / Pointers| Contract

    Contract -.->|Injected via DI| Default
    Contract -.->|Hot Swappable| Custom
```

📖 Gory Details: For the exact memory layout of the component storage and the incremental baking queues, read the CM Architectural Specification.

The Core Components
1. The Domain Layer: Where users define Topologies, Field Algebras, and Entities using rich, readable Python objects. It handles high-level rules without worrying about memory management.

2. The Component Manager (CM): The translation bridge.

Public API & Command Buffer: Safely accepts external data and inter-CM bridge data, queuing mutations to prevent race conditions.

The Translator: Strips away OOP bloat. It takes queued commands (bake_incremental) and flattens entity states and field weights into contiguous, strongly-typed memory blocks.

Component Storage: Maintains a synchronized dual-state: the mapped Python objects for user readouts, and the flat C-arrays for the execution engine.

3. The Kernel Contract (Dependency Injection): The rigid boundary that makes the engine modular. The Component Manager guarantees that the flat C-arrays will perfectly match a specific KernelDataContract.

4. The Kernel Layer: The hot-loop execution engine. Bypassing the Python GIL entirely, the kernel receives the C-arrays via Fast Reference pointers and computes the wave expansions and collapses at hardware speeds.

Zero-Friction Hot Swapping
Because the execution engine is injected via Dependency Injection (DI) and only interacts with the KernelDataContract, kernels are perfectly hot-swappable. A user can seamlessly swap out the default Numba kernel for a custom C++ or CUDA kernel without rewriting a single line of their Domain layer logic.
