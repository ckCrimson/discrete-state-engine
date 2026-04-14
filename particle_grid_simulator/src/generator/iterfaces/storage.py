from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from typing import Any, Tuple, Callable

from hpc_ecs_core.src.hpc_ecs_core.interfaces import IKernelStorage, KernelDataContract
from particle_grid_simulator.src.generator.domain.interfaces.field_generator import IFieldGeneratorData

from dataclasses import dataclass
from typing import Any

from hpc_ecs_core.src.hpc_ecs_core.interfaces import KernelDataContract


@dataclass(frozen=True)
class GeneratorKernelDataContract(KernelDataContract):
    """
    The strict, immutable blueprint required to allocate Numba memory
    and configure the C-level execution loop.

    Frozen=True guarantees that once this is passed to the Storage
    allocator, the dimensions can never be accidentally changed.
    """

    # --- MEMORY BOUNDS (The "How Much") ---
    maximum_steps: int
    max_active_states: int

    # --- SHAPES & DIMENSIONS (The "What Size") ---
    state_dimensions: int
    input_field_size: int
    global_field_size: int

    # --- EXECUTION FLAGS ---
    intrinsic_norm: bool
    """Normalizes the transition field F_a after multiplication with the Global Field."""

    extrinsic_norm: bool
    """Conserves the total field magnitude across the entire frontier after a step."""

    @classmethod
    def from_domain(
            cls,
            generator_data: Any,  # Typed to your generic markovian data
            global_field_dim: int
    ) -> 'GeneratorKernelDataContract':
        """
        FACTORY: The firewall between OOP and C.
        Extracts primitive types from the rich domain objects.
        """

        # 1. Safely unwrap tuple shapes into flat integer dimensions
        # Example: state_shape (2,) becomes state_dimensions = 2
        s_dim = cls._unpack_shape(generator_data.state_shape, "state_shape")

        # Depending on how your domain handles this property name,
        # ensure it matches exactly what GenericMarkovianFieldGeneratorData exposes.
        # It's usually field_vector_shape or similar based on your algebra.
        f_dim = cls._unpack_shape(generator_data.field_vector_shape, "field_vector_shape")

        # 2. Extract, map, and freeze
        return cls(
            maximum_steps=generator_data.maximum_step_baking,
            max_active_states=generator_data.max_size,

            state_dimensions=s_dim,
            input_field_size=f_dim,
            global_field_size=global_field_dim,

            # ---> THE FIX: Map the Domain's old names to the Engine's physical names <---
            intrinsic_norm=generator_data.implicit_norm,
            extrinsic_norm=generator_data.explicit_norm
        )

    @staticmethod
    def _unpack_shape(shape_tuple: tuple, name: str) -> int:
        if not shape_tuple or len(shape_tuple) != 1:
            raise ValueError(f"{name} must be a 1D tuple, got {shape_tuple}")
        return shape_tuple[0]


@dataclass
class GeneratorKernelFastRef:
    # --- ACTIVE STATE (Ping-Pong Buffers) ---
    buffer_A_states: np.ndarray
    buffer_A_fields: np.ndarray
    buffer_B_states: np.ndarray
    buffer_B_fields: np.ndarray

    active_count_A: int
    active_count_B: int

    # --- TOPOLOGY ENVIRONMENT (CSR Arrays) ---
    state_coordinates: np.ndarray
    edge_offsets: np.ndarray
    edge_targets: np.ndarray

    # --- GLOBAL FIELD ENVIRONMENT ---
    global_states: np.ndarray
    global_fields: np.ndarray
    global_normalized_fields: np.ndarray

    # --- INJECTED ALGEBRA (The Fix) ---
    # Default to None so the Storage allocator doesn't need to provide them
    math_multiply: Callable = None
    math_norm: Callable = None


class IGeneratorKernelStorage(IKernelStorage):
    """
    UNIVERSAL CONTRACT: The minimum hardware memory required to run a Ping-Pong
    field generation step. Strictly manages the memory it *owns* (Buffer A & B).
    """

    @property
    @abstractmethod
    def fast_refs(self) -> 'GeneratorKernelFastRef':
        """Strongly typed FastRef for the Generator."""
        pass

    # ==========================================
    # OWNED MEMORY (Ping-Pong Buffers)
    # ==========================================
    @property
    @abstractmethod
    def buffer_A_states(self) -> np.ndarray: pass

    @property
    @abstractmethod
    def buffer_A_fields(self) -> np.ndarray: pass

    @property
    @abstractmethod
    def buffer_B_states(self) -> np.ndarray: pass

    @property
    @abstractmethod
    def buffer_B_fields(self) -> np.ndarray: pass


class ICSRGeneratorStorage(IGeneratorKernelStorage):
    """
    SPECIFIC CONTRACT: Hardware memory layout for kernels that traverse graphs
    using Compressed Sparse Row (CSR) arrays and utilize Global Field contexts.

    Note: The Storage class does NOT allocate this memory. It merely exposes
    the pointers that the Translator will inject into the FastRef.
    """

    # ==========================================
    # INJECTED MEMORY (Topology Environment)
    # ==========================================
    @property
    @abstractmethod
    def state_coordinates(self) -> np.ndarray: pass

    @property
    @abstractmethod
    def edge_offsets(self) -> np.ndarray: pass

    @property
    @abstractmethod
    def edge_targets(self) -> np.ndarray: pass

    # ==========================================
    # INJECTED MEMORY (Global Field Environment)
    # ==========================================
    @property
    @abstractmethod
    def global_states(self) -> np.ndarray: pass

    @property
    @abstractmethod
    def global_fields(self) -> np.ndarray: pass

    @property
    @abstractmethod
    def global_normalized_fields(self) -> np.ndarray: pass