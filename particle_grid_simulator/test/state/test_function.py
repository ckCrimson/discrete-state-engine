import time
import numpy as np

from hpc_ecs_core.src.hpc_ecs_core.interfaces import KernelDataContract
from particle_grid_simulator.src.state.component_manager.component_manager import StateComponentManager
from particle_grid_simulator.src.state.kernel.numba.storage.storage_v1 import NumbaStateStorage
from particle_grid_simulator.src.state.kernel.numba.translator.translator_v1 import NumbaStateTranslator
from particle_grid_simulator.src.state.kernel.numba.utility.utility_v1 import NumbaStateUtility


# ... (Previous Imports) ...

def test_run_pure_stress_test():
    print("💎 STARTING PURE STATE MANIFOLD STRESS TEST")
    print("-" * 60)

    # 1. Setup
    contract = KernelDataContract(max_count=1100000, dimensions=3)
    manager = StateComponentManager(
        contract=contract,
        storage=NumbaStateStorage(contract),
        translator=NumbaStateTranslator(),
        utility=NumbaStateUtility()
    )

    # ==========================================
    # 🔥 PHASE 0: THE WARM-UP (JIT Compilation)
    # ==========================================
    print("--> Warming up Numba Kernels (Compiling)...")
    warmup_data = {
        'ids': np.array([9999999], dtype=np.int64),
        'coords': np.array([[0, 0, 0]], dtype=np.float32)
    }
    # These calls will take ~4 seconds the first time, but 0.0001s thereafter
    manager.union_in_place(warmup_data)
    manager.intersection_in_place(warmup_data)
    # Clean the slate after warmup
    manager.clear()
    print("✅ Kernels Cached. Moving to high-speed execution.\n")

    # ==========================================
    # 🚀 PHASE 1: THE MILLION STATE RUN
    # ==========================================
    print("--> Generating 1,000,000 3D coordinates...")
    raw_coords = np.random.uniform(-100, 100, (1000000, 3)).astype(np.float32)
    raw_ids = np.arange(1, 1000001, dtype=np.int64)
    raw_init_data = {'ids': raw_ids, 'coords': raw_coords}

    # Time Initialization
    start = time.perf_counter()
    manager.add_state(raw_init_data)
    manager.commit_frame()
    print(f"✅ 1M States Initialized: {(time.perf_counter() - start) * 1000:.2f} ms")

    # Time Mapping (3D to 2D)
    def project_to_2d(ids_arr, coords_arr):
        return coords_arr[:, :2]

    print("\n--> Mapping: Projecting 1M States from 3D to 2D...")
    start = time.perf_counter()
    manager.map_in_place(project_to_2d)
    print(f"✅ Mapping (3D->2D) Complete: {(time.perf_counter() - start) * 1000:.2f} ms")

    # ==========================================
    # ⚖️ PHASE 2: SET OPERATIONS
    # ==========================================
    print("\n--> Performing Set Operations (100 states, 60 overlap)...")
    test_ids = np.concatenate([np.arange(1, 61), np.arange(2000001, 2000041)])
    test_coords = np.random.uniform(-100, 100, (100, 2)).astype(np.float32)
    test_batch = {'ids': test_ids, 'coords': test_coords}

    # 1. UNION TEST
    # Starts at 1,000,000. Adds 40 new ones.
    manager.union_in_place(test_batch)
    union_count = np.sum(manager.fast_refs['active_mask'])
    print(f"✅ Union Count: {union_count} (Expected 1000040)")

    # 2. INTERSECTION TEST
    # To test a clean intersection, let's see how many of the 1,000,040
    # match our original 100-count batch.
    start = time.perf_counter()
    manager.intersection_in_place(test_batch)
    print(f"✅ Intersection (Filter 1M down to 100): {(time.perf_counter() - start) * 1000:.2f} ms")

    # 3. Final Result Verification
    active_count = manager.count
    print(f"\n📊 Final Active Count: {active_count}")

    # Logic: Intersection should now be 100, because all 100 IDs in
    # 'test_batch' are now present in the manager (60 from start + 40 from Union).
    assert active_count == 100