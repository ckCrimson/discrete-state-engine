from typing import Dict, Any, Iterable, Type
import numpy as np

from particle_grid_simulator.src.topology.interfaces.translator import ITopologyTranslator


class NumbaTopologyTranslator(ITopologyTranslator):

    def to_raw_vector(self, state_obj: Any) -> np.ndarray:
        """Strips the high-level Python object down to its raw NumPy vector."""
        vec = state_obj.vector

        # GATEKEEPER: Ensure the kernel always gets a NumPy array
        if isinstance(vec, np.ndarray):
            return vec
        return np.array(vec, dtype=np.int32)

    def to_state_objects(self, raw_vectors: Iterable[np.ndarray], state_class: Type) -> Iterable[Any]:
        """Wraps raw NumPy arrays back into the user's State class."""
        return [state_class(vec) for vec in raw_vectors]

    def bake(self, fast_refs: Dict[str, Any], initial_data: Any) -> None:
        """
        Extracts starting states from the high-level Topology object
        and injects them into the Numba-optimized memory structures.
        """
        if initial_data is None:
            return

        topo_ref = fast_refs["topology"]
        topology = initial_data  # initial_data is the Topology object passed from the Manager

        # Assuming the topology object exposes its initial states via a property or method
        # Adjust 'topology.states' to match your actual domain object's attribute
        starting_states = getattr(topology, 'states', [])

        for state_obj in starting_states:
            raw_vec = self.to_raw_vector(state_obj)
            vec_tuple = tuple(raw_vec.tolist())

            # Sparse Set initialization: Map and Array
            if vec_tuple not in topo_ref.visited_map:
                new_idx = len(topo_ref.handle_map)
                topo_ref.visited_map[vec_tuple] = new_idx
                topo_ref.handle_map.append(raw_vec)



    def sync(self, fast_refs: Dict[str, Any], **kwargs) -> None:
        """
        Hydrates the high-level Topology domain object's adjacency_cache
        using the raw Numba Dynamic CSR graph.
        """
        topology = kwargs.get("topology")

        if not topology or not getattr(topology, 'use_cache', False):
            return

        topo_ref = fast_refs["topology"]
        state_class = topology.state_class

        total_states = len(topo_ref.handle_map)
        if total_states == 0:
            return

        # 1. OPTIMIZATION: Create Singleton State objects to avoid memory thrashing.
        state_instances = [None] * total_states
        for i in range(total_states):
            state_instances[i] = state_class(topo_ref.handle_map[i])

        # 2. Reconstruct the graph cache
        for src_idx in range(total_states):
            count = topo_ref.forward_counts[src_idx]

            if count == 0:
                continue

            src_state = state_instances[src_idx]

            # Only sync if not already in the domain cache
            if src_state not in topology.adjacency_cache:
                start = topo_ref.forward_starts[src_idx]

                neighbors = tuple(
                    state_instances[topo_ref.forward_edges[start + i]]
                    for i in range(count)
                )

                topology.adjacency_cache[src_state] = neighbors