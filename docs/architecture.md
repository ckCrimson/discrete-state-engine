# Component Manager (CM) Architectural Specification

## 1. Overview
The FDS Component Manager (CM) is the central orchestration layer of the Dual-Flow Architecture. It is strictly responsible for translating human-readable Domain logic into hardware-optimized memory layouts (Data-Oriented Design) and managing memory synchronization during the execution hot-loop.

The CM ensures that the execution Kernel is completely decoupled from the Domain layer through strict Dependency Injection (DI).

## 2. Initialization and The Contract
The CM does not inherently know what backend will be executing the simulation. It relies on a `KernelDataContract` passed into its constructor.

* **The Contract:** Defines the exact contiguous memory arrays (e.g., C-arrays, CSR matrices) required by the execution backend.
* **Component Storage (CS):** Upon instantiation, the CM's constructor uses the Contract to allocate an empty Component Storage block. This storage holds the dual-state data: the mapped Python objects for Domain reading, and the flat memory arrays for hardware execution.

## 3. The Translator Layer
The Translator is the CM's internal serialization engine. It exposes three strict operations to move data between the Domain and the Component Storage:

1. **`bake()` (Full Bake):** A high-cost, one-time operation used during setup. It traverses the active Domain objects (Entities, Topologies, Fields) and flattens them into the allocated Component Storage arrays.
2. **`sync()`:** Synchronizes the results from the hardware arrays back into the mapped Domain objects, allowing the user to read the results of a simulation step in rich Python.
3. **`bake_incremental()`:** The hot-loop operation. Instead of full traversal, the CM reads from a **Command Buffer** and pushes only specific, queued mutations into the Component Storage at C-speeds.

## 4. Inter-CM Communication (Bridge Data)
In multi-component simulations (e.g., combining a Topology CM with a Field CM), managers must communicate without breaking the hot-loop.
* The CM exposes a **Public API** that accepts `Bridge Data`.
* This data is queued into the Command Buffer and processed during the next `bake_incremental` cycle, ensuring thread-safe, race-condition-free mutations.

## 5. Fast References and The Utility Layer
To achieve C-level speeds in Python, the CM entirely bypasses standard object lookups using **Fast References** (raw pointers or memoryviews to the Component Storage arrays).

* **Internal Fast References:** The CM passes a Fast Reference to its internal **Utility API**. This allows `Utility Methods` to perform rapid calculations (e.g., finding topological neighbors) by reading directly from the arrays without waiting for a `sync()`.
* **External Fast References:** The Component Storage provides an external Fast Reference to the injected Execution Kernel (e.g., a Numba `@njit` loop or a C++ binary). The Kernel reads and writes directly to this memory block, bypassing the Python Global Interpreter Lock (GIL).

## 6. Hot-Swapping the Kernel
Because the Execution Kernel only interacts with the externally provided Fast Reference and the `KernelDataContract`, the CM is entirely agnostic to the execution backend. Swapping a Numba kernel for a custom CUDA kernel requires zero changes to the Component Manager or the Domain logic.