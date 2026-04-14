# Particle Field Simulation Engine

A highly scalable, physics-agnostic, and hardware-agnostic simulation framework built on an Entity Component System (ECS) architecture.

## Overview

This engine is designed to simulate any arbitrary set of particles under user-defined fields and interaction functions. Instead of hardcoding specific physics (like fluid dynamics or gravity), the engine provides a robust mathematical and spatial foundation where custom field generation logic can be injected.

Furthermore, the engine is completely decoupled from its execution hardware. Through a strict Strategy/Adapter pattern, the high-level domain logic is seamlessly translated into low-level, memory-aligned structures for processing on dynamically selected kernels (CPU, CUDA, OpenCL, etc.), depending on the best fit for the specific application.

## Core Architecture

The engine enforces strict module encapsulation to ensure maintainability and high performance:

* **The Component Manager (Facade):** The sole public interface of the engine. Users interact strictly with the `SimulationComponentManager` to register particles, define fields, and step the simulation.
* **Domain Logic:** High-level abstractions defining the spatial hashing, interaction rules, and arbitrary field properties.
* **Kernels:** Hardware-specific implementations. Each kernel contains its own raw storage structures, utility functions, and high-to-low data translators to guarantee optimal memory alignment (e.g., Array-of-Structures to Structure-of-Arrays conversions for GPU SIMD execution).

## Directory Structure

```text
simulation_engine/
├── src/
│   └── simulation_engine/
│       ├── __init__.py           # Exposes ONLY the SimulationComponentManager
│       ├── manager.py            # Facade implementation
│       ├── _domain/              # Private: High-level mathematical/physics logic
│       └── _kernels/             # Private: Hardware specific execution strategies
│           ├── _cpu/
│           └── _cuda/
└── tests/                        # Mirrored test suite
```

## Installation

*(Note: This package requires the `hpc_ecs_core` to be installed in your environment).*

To install the engine for local development with testing dependencies:

```bash
pip install -e .[test]
```

## Usage Example

```python
from simulation_engine import SimulationComponentManager

# Initialize the simulation facade
sim = SimulationComponentManager()

# The manager abstracts away the underlying kernel execution and memory translation
sim.initialize_environment(...)
sim.step()
```

## Testing

This project utilizes `pytest` to ensure robust validation across all kernels and domain logic. To run the full test suite:

```bash
pytest tests/
```