from typing import List, Any, Set, Tuple

import numpy as np

from particle_grid_simulator.src.state.component_manager.component_enums import CommandType
from particle_grid_simulator.src.state.domain.state_domain import State
from particle_grid_simulator.src.state.interfaces.storage import StateFastRefs
from particle_grid_simulator.src.state.interfaces.translator import IStateTranslator


class NumbaStateTranslator(IStateTranslator):
    def bake(self, fast_refs: StateFastRefs, initial_data: Any = None) -> None:
        if not initial_data:
            return

        # Fast Path: Raw dict
        if isinstance(initial_data, dict) and 'ids' in initial_data:
            count = len(initial_data['ids'])
            fast_refs['ids'][:count] = initial_data['ids']
            fast_refs['active_mask'][:count] = 1
            fast_refs['coords'][:count] = initial_data['coords']
            return

        # Slow Path: State Objects
        vectors = np.array([state.vector for state in initial_data])
        count = len(vectors)
        if count > 0:
            fast_refs['ids'][:count] = vectors[:, 0]
            fast_refs['active_mask'][:count] = 1
            # Coords start from index 1 to the end
            fast_refs['coords'][:count] = vectors[:, 1:]

    def bake_incremental(self, fast_refs: StateFastRefs, command_queue: List[Tuple[int, str, Any]], **kwargs) -> None:
        for _, cmd_name, data in command_queue:
            if cmd_name == 'ADD_BATCH':
                avail = np.where(fast_refs['active_mask'] == 0)[0]

                # 1. Fast Path: Raw Dictionary
                if isinstance(data, dict) and 'ids' in data:
                    new_ids = data['ids']
                    new_coords = data['coords']
                    count = len(new_ids)

                # 2. Slow Path: State Objects or Iterables
                else:
                    vectors = np.array([state.vector if isinstance(state, State) else state for state in data])
                    count = len(vectors)
                    if count == 0:
                        continue
                    new_ids = vectors[:, 0]
                    new_coords = vectors[:, 1:]

                # 3. Insert into hardware arrays
                if count <= len(avail):
                    target_idx = avail[:count]
                    fast_refs['ids'][target_idx] = new_ids
                    fast_refs['active_mask'][target_idx] = 1
                    fast_refs['coords'][target_idx] = new_coords
                else:
                    raise MemoryError("StateStorage capacity exceeded.")

            elif cmd_name == CommandType.DELETE_BATCH.name:
                # Add dict safeguard here too just in case!
                if isinstance(data, dict) and 'ids' in data:
                    delete_ids = data['ids']
                else:
                    delete_ids = np.array([s.vector[0] if isinstance(s, State) else s[0] for s in data])

                mask = np.isin(fast_refs['ids'], delete_ids) & (fast_refs['active_mask'] == 1)
                fast_refs['active_mask'][mask] = 0

    def sync(self, fast_refs: StateFastRefs, **kwargs) -> Set[State]:
        active = np.where(fast_refs['active_mask'] == 1)[0]
        if len(active) == 0: return set()

        ids = fast_refs['ids'][active]
        coords = fast_refs['coords'][active]

        synced = set()
        for i in range(len(active)):
            # Unified vector: [ID, x, y, z...]
            vec = np.concatenate(([ids[i]], coords[i]))
            synced.add(State(vector=vec))
        return synced