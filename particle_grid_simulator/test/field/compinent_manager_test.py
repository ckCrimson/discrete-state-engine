import numpy as np

from particle_grid_simulator.src.field.component_manager.component_manager import FieldComponentManager
from particle_grid_simulator.src.field.domain.data.field_algebra import FieldAlgebra
from particle_grid_simulator.src.field.domain.data.field_mapper import FieldMapper
from particle_grid_simulator.src.field.interfaces.storage import FieldKernelDataContract
from particle_grid_simulator.src.field.kernel.numba.storage.storage_v1 import NumbaFieldKernelStorage
from particle_grid_simulator.src.field.kernel.numba.translator.translator_v1 import NumbaFieldTranslator
from particle_grid_simulator.src.field.kernel.numba.utility.utility_v1 import NumbaKernelFieldUtility
from particle_grid_simulator.src.state.domain import State


# Adjust these imports based on your actual project structure
# from particle_grid_simulator.src.field.domain.algebra import FieldAlgebra
# from particle_grid_simulator.src.field.domain.mappers import FieldMapper
# from particle_grid_simulator.src.field.kernel.contracts import FieldKernelDataContract
# from particle_grid_simulator.src.field.kernel.storage.numba_storage import NumbaFieldKernelStorage
# from particle_grid_simulator.src.field.kernel.transformation.numba_translator import NumbaFieldTranslator
# from particle_grid_simulator.src.field.kernel.utility.vector_utility import VectorFieldKernelUtility
# from particle_grid_simulator.src.field.component_manager.manager import FieldComponentManager

def test_field_component_manager_api():
    print("--- STARTING FIELD COMPONENT MANAGER INTEGRATION TEST ---")

    # ==========================================
    # 1. SETUP DOMAIN
    # ==========================================
    print("\n1. Setting up Domain...")
    # Using a simple 2D Vector Algebra
    algebra = FieldAlgebra(dimensions=2, dtype=np.float64)

    # Define 3 specific points in space
    state_a =  np.array([0.0, 0.0])
    state_b = np.array([1.0, 1.0])
    state_c = np.array([2.0, 2.0])
    states = [state_a, state_b, state_c]

    # Define initial vectors pointing at [10, 10]
    initial_fields = [np.array([10.0, 10.0]) for _ in states]

    # Initialize the pure Python Mapper
    domain_mapper = FieldMapper(
        algebra=algebra,
        state_class_ref=State,  # Using raw arrays as 'States' for the test
        states=states,
        field_vectors=initial_fields
    )


    # ==========================================
    # 2. SETUP HARDWARE COMPONENT MANAGER
    # ==========================================
    print("2. Spinning up Hardware Manager...")
    contract = FieldKernelDataContract(state_dimensions=2, field_dimensions=2, initial_capacity=10, algebra=FieldAlgebra(dimensions=2)
                                       ,state_class_ref=State)
    storage = NumbaFieldKernelStorage(contract)
    translator = NumbaFieldTranslator()
    utility = NumbaKernelFieldUtility()  # Assuming the renamed Vector utility

    manager = FieldComponentManager(
        contract=contract,
        storage=storage,
        translator=translator,
        utility=utility,
        domain_mapper=domain_mapper
    )


    # ==========================================
    # 3. TEST GET_FIELDS (Initial Bake Verification)
    # ==========================================
    print("3. Testing get_fields() & Bake...")
    out_fields = manager.get_fields(states)

    assert out_fields.shape == (3, 2), "Shape mismatch on retrieval."
    assert np.all(out_fields == 10.0), "Initial baked data is incorrect."
    print("   -> Passed!")

    # ==========================================
    # 4. TEST SET_FIELDS & DEFERRED EXECUTION
    # ==========================================
    print("4. Testing set_fields() & Deferred Queue...")
    # Overwrite State A to [99.0, 99.0]
    manager.set_fields(states=[state_a], fields=[np.array([99.0, 99.0])])

    # Pre-commit check: Should STILL be 10.0
    pre_commit = manager.get_fields([state_a])
    assert np.all(pre_commit == 10.0), "Data mutated before commit_frame()!"

    manager.commit_frame()

    # Post-commit check: Should be 99.0
    post_commit = manager.get_fields([state_a])
    assert np.all(post_commit == 99.0), "SET command failed during commit."
    print("   -> Passed!")

    # ==========================================
    # 5. TEST ADD_FIELDS
    # ==========================================
    print("5. Testing add_fields()...")
    # Add [5.0, 5.0] to State B (Currently 10.0)
    manager.add_fields(states=[state_b], fields=[np.array([5.0, 5.0])])
    manager.commit_frame()

    post_add = manager.get_fields([state_b])
    assert np.all(post_add == 15.0), f"ADD command failed. Expected 15.0, got {post_add[0]}"
    print("   -> Passed!")

    # ==========================================
    # 6. TEST CLEAR_FIELDS
    # ==========================================
    print("6. Testing clear_fields()...")
    # Clear State C (Currently 10.0) back to algebraic null [0.0, 0.0]
    manager.clear_fields(states=[state_c])
    manager.commit_frame()

    post_clear = manager.get_fields([state_c])
    assert np.all(post_clear == 0.0), f"CLEAR command failed. Expected 0.0, got {post_clear[0]}"
    print("   -> Passed!")

    # ==========================================
    # 7. TEST SYNC_TO_DOMAIN
    # ==========================================
    print("7. Testing sync_to_domain()...")
    manager.sync_to_domain()

    # The OOP Domain cache should now reflect the hardware mutations:
    # A = [99.0, 99.0], B = [15.0, 15.0], C = [0.0, 0.0]

    a_tuple = tuple(state_a.tolist())
    b_tuple = tuple(state_b.tolist())
    c_tuple = tuple(state_c.tolist())

    cache = domain_mapper.mapping_cache
    assert np.all(cache[a_tuple] == 99.0), "Sync failed for SET operation."
    assert np.all(cache[b_tuple] == 15.0), "Sync failed for ADD operation."
    assert np.all(cache[c_tuple] == 0.0), "Sync failed for CLEAR operation."

    print("   -> Passed!")
    print("\n--- ALL COMPONENT MANAGER TESTS PASSED ---")


if __name__ == "__main__":
    test_field_component_manager_api()