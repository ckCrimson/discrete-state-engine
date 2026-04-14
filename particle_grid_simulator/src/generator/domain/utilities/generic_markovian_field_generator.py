import numpy as np
from typing import List, Callable, Dict, Tuple, Any

from particle_grid_simulator.src.field.domain.interfaces.mapper_interface import IFieldMapper
from particle_grid_simulator.src.generator.domain.interfaces.field_generator import IFieldGeneratorData
from particle_grid_simulator.src.topology.domain.utility.utility import TopologyUtility


# Assuming these are imported from your domain interfaces
# from particle_grid_simulator.src.field.domain.interfaces.generator_interface import IFieldGeneratorUtility, IFieldGeneratorData
# from particle_grid_simulator.src.field.domain.interfaces.mapper_interface import IFieldMapper
# from particle_grid_simulator.src.topology.domain.utility.utility import TopologyUtility

class GenericMarkovianFieldGeneratorUtility:
    """
    UTILITY: Pure mathematical execution of the Generic Markovian Field Generator.
    Stateless, decoupled, and heavily optimized for native Python via variable hoisting,
    Ping-Pong buffers, and C-level byte hashing.
    """

    @staticmethod
    def calculate_affected_transition_field(
            source_state: np.ndarray,
            target_state: np.ndarray,
            global_field_vector: np.ndarray,
            transition_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            multiply_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Calculates F_a(s_j -> s_i) = T(s_j -> s_i) ⊗ F_g(s_i)
        """
        raw_transition = transition_func(source_state, target_state)
        return multiply_func(raw_transition, global_field_vector)

    @staticmethod
    def normalize_transition_frontier(
            affected_fields: List[np.ndarray],
            add_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            norm_func: Callable[[np.ndarray], Any]
    ) -> List[np.ndarray]:
        """
        Calculates Z_i over the reachable space (PDF Normalization).
        """
        if not affected_fields:
            return []

        total_field = affected_fields[0]
        for f in affected_fields[1:]:
            total_field = add_func(total_field, f)

        total_norm = norm_func(total_field)

        if total_norm > 0:
            return [f / total_norm for f in affected_fields]
        else:
            return [np.zeros_like(f) for f in affected_fields]

    @staticmethod
    def _generate_next_step(
            current_states: np.ndarray,
            current_fields: np.ndarray,
            active_count: int,
            target_states_buffer: np.ndarray,
            target_fields_buffer: np.ndarray,
            global_mapper: 'IFieldMapper',
            generator_data: 'IFieldGeneratorData',
            do_implicit_norm: bool,
            do_explicit_norm: bool,
            **kwargs: Any
    ) -> int:
        """
        PRIVATE: Executes a single generation step.
        Reads from current arrays and accumulates into target pre-allocated buffers.
        """
        # ==========================================
        # 1. VARIABLE HOISTING (Performance Crux)
        # ==========================================
        t_func = generator_data.transition_function
        mul_func = generator_data.algebra_multiply
        add_func = generator_data.algebra_add
        norm_func = generator_data.algebra_norm
        unity_vec = generator_data.algebra_unity_vector
        null_vec = generator_data.algebra_null_vector

        topology = getattr(generator_data, 'topology')
        state_class = generator_data.state_class_ref

        # OPTIMIZATION 1: Fast C-level byte hashing for array accumulation
        state_to_index: Dict[bytes, int] = {}
        next_active_count = 0

        # OPTIMIZATION 2: Pre-extract global field into a fast dictionary
        g_states, g_fields = global_mapper.get_raw_data()
        global_dict = {s.tobytes(): f for s, f in zip(g_states, g_fields)}

        # ==========================================
        # 2. THE BLAST LOOP
        # ==========================================
        for i in range(active_count):
            s_j_vec = current_states[i]
            current_field_s_j = current_fields[i]

            # Hydrate to object strictly for the discrete Topology cache lookup
            s_j_obj = state_class(s_j_vec)
            neighbors = TopologyUtility._get_cached_reachable(topology, s_j_obj)

            if not neighbors:
                continue

            affected_fields_for_frontier = []
            target_vecs = []

            # Step A: Modulate transitions by the Global Environment
            for s_i_obj in neighbors:
                s_i_vec = s_i_obj.vector
                target_vecs.append(s_i_vec)

                # OPTIMIZATION 2 (Applied): Instant byte lookup, no tuple conversion
                f_g = global_dict.get(s_i_vec.tobytes())
                if f_g is None:
                    f_g = unity_vec

                # Calculate raw affected field
                f_a = GenericMarkovianFieldGeneratorUtility.calculate_affected_transition_field(
                    s_j_vec, s_i_vec, f_g, t_func, mul_func
                )

                # OPTIONAL: IMPLICIT NORM
                if do_implicit_norm:
                    mag = norm_func(f_a)
                    if mag > 0:
                        f_a = f_a / mag
                    else:
                        f_a = np.zeros_like(f_a)

                affected_fields_for_frontier.append(f_a)

            # Step B: Conserve the field over the frontier
            if do_explicit_norm:
                normalized_fields = GenericMarkovianFieldGeneratorUtility.normalize_transition_frontier(
                    affected_fields_for_frontier, add_func, norm_func
                )
            else:
                normalized_fields = affected_fields_for_frontier

            # Step C: Accumulate directly into the target NumPy buffers
            for s_i_vec, f_s in zip(target_vecs, normalized_fields):
                # OPTIMIZATION 1 (Applied): Bypass tuple(tolist()) penalty entirely
                s_i_bytes = s_i_vec.tobytes()
                contribution = mul_func(current_field_s_j, f_s)

                idx = state_to_index.get(s_i_bytes, -1)

                if idx == -1:
                    # New state discovered, allocate next row in buffer
                    idx = next_active_count
                    state_to_index[s_i_bytes] = idx
                    target_states_buffer[idx] = s_i_vec
                    target_fields_buffer[idx] = contribution
                    next_active_count += 1
                else:
                    # State exists, accumulate in place
                    target_fields_buffer[idx] = add_func(target_fields_buffer[idx], contribution)

        # ==========================================
        # 3. BUFFER SANITATION
        # ==========================================
        # Ensure the unused portion of the target buffer is zeroed out
        if next_active_count < target_fields_buffer.shape[0]:
            target_fields_buffer[next_active_count:] = null_vec
            target_states_buffer[next_active_count:] = 0.0

        return next_active_count

    @staticmethod
    def generate_multi_step_field(
            initial_mapper: 'IFieldMapper',
            global_mapper: 'IFieldMapper',
            generator_data: 'IFieldGeneratorData',
            steps: int,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        PUBLIC: Drives the L-step evolution.
        Manages the ping-pong buffer allocation and executes the loop.
        """
        if steps <= 0:
            raw_states, raw_fields = initial_mapper.get_raw_data()
            return np.array(raw_states), np.array(raw_fields)

        # Safely extract Markovian-specific rules
        do_implicit = getattr(generator_data, 'implicit_norm', False)
        do_explicit = getattr(generator_data, 'explicit_norm', False)

        max_size = generator_data.max_size
        s_shape = generator_data.state_shape
        f_shape = generator_data.field_vector_shape

        # ==========================================
        # 1. PING-PONG BUFFER ALLOCATION
        # ==========================================
        buffer_A_states = np.zeros((max_size, *s_shape), dtype=np.float64)
        buffer_A_fields = np.zeros((max_size, *f_shape), dtype=np.float64)

        buffer_B_states = np.zeros((max_size, *s_shape), dtype=np.float64)
        buffer_B_fields = np.zeros((max_size, *f_shape), dtype=np.float64)

        # ==========================================
        # 2. SEED INITIAL STATE
        # ==========================================
        raw_states, raw_fields = initial_mapper.get_raw_data()
        active_count = len(raw_states)

        if active_count > max_size:
            raise MemoryError("Initial active states exceed max_size generator capacity.")

        buffer_A_states[:active_count] = np.array(raw_states)
        buffer_A_fields[:active_count] = np.array(raw_fields)

        # ==========================================
        # 3. EXECUTION LOOP
        # ==========================================
        for l in range(steps):
            if l % 2 == 0:
                # Read A -> Write B
                active_count = GenericMarkovianFieldGeneratorUtility._generate_next_step(
                    current_states=buffer_A_states,
                    current_fields=buffer_A_fields,
                    active_count=active_count,
                    target_states_buffer=buffer_B_states,
                    target_fields_buffer=buffer_B_fields,
                    global_mapper=global_mapper,
                    generator_data=generator_data,
                    do_implicit_norm=do_implicit,
                    do_explicit_norm=do_explicit,
                    **kwargs
                )
            else:
                # Read B -> Write A
                active_count = GenericMarkovianFieldGeneratorUtility._generate_next_step(
                    current_states=buffer_B_states,
                    current_fields=buffer_B_fields,
                    active_count=active_count,
                    target_states_buffer=buffer_A_states,
                    target_fields_buffer=buffer_A_fields,
                    global_mapper=global_mapper,
                    generator_data=generator_data,
                    do_implicit_norm=do_implicit,
                    do_explicit_norm=do_explicit,
                    **kwargs
                )

            if active_count >= max_size:
                raise MemoryError("Generator max_size exceeded during multi-step expansion.")

        # ==========================================
        # 4. RETURN PACKED SLICES
        # ==========================================
        if steps % 2 == 1:
            return buffer_B_states[:active_count].copy(), buffer_B_fields[:active_count].copy()
        else:
            return buffer_A_states[:active_count].copy(), buffer_A_fields[:active_count].copy()