from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Set, Type

import numpy as np

from particle_grid_simulator.src.state.domain import State
from particle_grid_simulator.src.topology.domain.topology_domain import Topology
from particle_grid_simulator.src.topology.domain.utility.utility import TopologyUtility


# ==========================================
# 2. THE TEST CASE (1D Random Walk)
# ==========================================
def test_1d_topology():
    print("🧪 Running 1D Topology Domain Test...\n")

    # 1. Define the rules of the universe (Using NumPy arrays!)
    def move_1d(s: State) -> Iterable[State]:
        x = s.vector[0]
        # Return new states with numpy arrays
        return [State(np.array([x - 1])), State(np.array([x + 1]))]

    # 2. Instantiate the Topology
    universe = Topology(reachable_func=move_1d, state_class=State)

    # 3. Define our starting point (Origin)
    s_0 = State(np.array([0]))

    # --- TEST 1: Immediate Reachable (l=1) ---
    step_1 = TopologyUtility.get_reachable(universe, s_0)
    expected_1 = {State(np.array([-1])), State(np.array([1]))}
    assert step_1.states == expected_1, f"Failed Reachable! Got {step_1.states}"
    print(f"✅ Immediate Reachable V(0) passed.")

    # --- TEST 2: Multi-Step Frontier (l=2) ---
    frontier_2 = TopologyUtility.get_multi_step_reachable_frontier(universe, s_0, l=2)
    expected_f2 = {State(np.array([-2])), State(np.array([0])), State(np.array([2]))}
    assert frontier_2.states == expected_f2, f"Failed Frontier! Got {frontier_2.states}"
    print(f"✅ 2-Step Frontier V^2(0) passed.")

    # --- TEST 3: Multi-Step Basin (l=2) ---
    basin_2 = TopologyUtility.get_multi_step_reachable_basin(universe, s_0, l=2)
    expected_b2 = {
        State(np.array([-2])), State(np.array([-1])), State(np.array([0])),
        State(np.array([1])), State(np.array([2]))
    }
    assert basin_2.states == expected_b2, f"Failed Basin! Got {basin_2.states}"
    print(f"✅ 2-Step Basin U V^i(0) passed.")

    print("\n🎉 All domain logic tests passed successfully! The OOP Ground Truth is solid.")
if __name__ == "__main__":
    test_1d_topology()