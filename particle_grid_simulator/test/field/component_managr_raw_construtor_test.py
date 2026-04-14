import time

import numpy as np

# Adjust imports to match your project structure
from particle_grid_simulator.src.field.component_manager.component_manager import FieldComponentManager
from particle_grid_simulator.src.field.domain.data.field_algebra import FieldAlgebra
from particle_grid_simulator.src.field.interfaces.storage import FieldKernelDataContract
from particle_grid_simulator.src.field.kernel.numba.storage.storage_v1 import NumbaFieldKernelStorage
from particle_grid_simulator.src.field.kernel.numba.translator.translator_v1 import NumbaFieldTranslator
from particle_grid_simulator.src.field.kernel.numba.utility.utility_v1 import NumbaKernelFieldUtility
from particle_grid_simulator.src.state.domain import State


def test_field_component_manager_raw_api():
    print("--- STARTING FIELD COMPONENT MANAGER RAW API TEST ---")

    # ==========================================
    # 1. SETUP RAW DATA (Bypassing the Mapper)
    # ==========================================
    print("\n1. Setting up Raw C-Arrays...")
    algebra = FieldAlgebra(dimensions=2, dtype=np.float64)

    # Define 3 specific points in space
    state_a = np.array([0.0, 0.0])
    state_b = np.array([1.0, 1.0])
    state_c = np.array([2.0, 2.0])
    raw_states = [state_a, state_b, state_c]

    # Initial vectors pointing at [10, 10]
    raw_fields = [np.array([10.0, 10.0]) for _ in raw_states]

    # ==========================================
    # 2. SETUP HARDWARE (VIA RAW CONSTRUCTOR)
    # ==========================================
    print("2. Spinning up Hardware Manager via create_from_raw()...")
    contract = FieldKernelDataContract(
        state_dimensions=2,
        field_dimensions=2,
        initial_capacity=10,
        algebra=algebra,
        state_class_ref=State
    )
    storage = NumbaFieldKernelStorage(contract)
    translator = NumbaFieldTranslator()
    utility = NumbaKernelFieldUtility()

    manager = FieldComponentManager.create_from_raw(
        contract=contract,
        storage=storage,
        translator=translator,
        utility=utility,
        states=raw_states,
        fields=raw_fields
    )


    # ==========================================
    # 3. TEST GET_FIELDS (Initial Bake Verification)
    # ==========================================
    print("3. Testing get_fields() & Initial Raw Bake...")
    out_fields = manager.get_fields(raw_states)

    assert out_fields.shape == (3, 2), "Shape mismatch on retrieval."
    assert np.all(out_fields == 10.0), "Initial baked data is incorrect."
    print("   -> Passed!")

    # ==========================================
    # 4. TEST SET_FIELDS & DEFERRED EXECUTION
    # ==========================================
    print("4. Testing set_fields() & Deferred Queue...")
    manager.set_fields(states=[state_a], fields=[np.array([99.0, 99.0])])

    # Pre-commit check
    pre_commit = manager.get_fields([state_a])
    assert np.all(pre_commit == 10.0), "Data mutated before commit_frame()!"

    manager.commit_frame()

    # Post-commit check
    post_commit = manager.get_fields([state_a])
    assert np.all(post_commit == 99.0), "SET command failed during commit."
    print("   -> Passed!")

    # ==========================================
    # 5. TEST ADD_FIELDS
    # ==========================================
    print("5. Testing add_fields()...")
    manager.add_fields(states=[state_b], fields=[np.array([5.0, 5.0])])
    manager.commit_frame()

    post_add = manager.get_fields([state_b])
    assert np.all(post_add == 15.0), f"ADD command failed. Expected 15.0, got {post_add[0]}"
    print("   -> Passed!")

    # ==========================================
    # 6. TEST CLEAR_FIELDS
    # ==========================================
    print("6. Testing clear_fields()...")
    manager.clear_fields(states=[state_c])
    manager.commit_frame()

    post_clear = manager.get_fields([state_c])
    assert np.all(post_clear == 0.0), f"CLEAR command failed. Expected 0.0, got {post_clear[0]}"
    print("   -> Passed!")

    # ==========================================
    # 7. TEST DYNAMIC REGISTRATION (NEW STATE)
    # ==========================================
    print("7. Testing dynamic registration of a brand new state...")
    # This state was never initialized in the raw_states array!
    state_d = np.array([5.0, 5.0])

    manager.set_fields(states=[state_d], fields=[np.array([42.0, 42.0])])
    manager.commit_frame()

    post_dynamic = manager.get_fields([state_d])
    assert np.all(post_dynamic == 42.0), "Dynamic registration in incremental bake failed."
    print("   -> Passed!")

    print("\n--- ALL RAW API TESTS PASSED SUCESSFULLY ---")


if __name__ == "__main__":
    test_field_component_manager_raw_api()