import time
import os
import numpy as np
import matplotlib.pyplot as plt

# Adjust imports to match your project structure
from particle_grid_simulator.src.field.component_manager.component_manager import FieldComponentManager
from particle_grid_simulator.src.field.domain.data.field_algebra import FieldAlgebra
from particle_grid_simulator.src.field.interfaces.storage import FieldKernelDataContract
from particle_grid_simulator.src.field.kernel.numba.storage.storage_v1 import NumbaFieldKernelStorage
from particle_grid_simulator.src.field.kernel.numba.translator.translator_v1 import NumbaFieldTranslator
from particle_grid_simulator.src.field.kernel.numba.utility.utility_v1 import NumbaKernelFieldUtility
from particle_grid_simulator.src.state.domain import State


def run_3d_benchmark():
    print("--- STARTING 3D HARDWARE BENCHMARK ---")

    # Target path for the plot
    plot_dir = r"E:\Particle Field Simulation\particle_grid_simulator\test\field\plot"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "benchmark_results.png")

    l_values = [1, 2, 3, 4, 50, 100]

    # 3D Vector Algebra
    algebra = FieldAlgebra(dimensions=3, dtype=np.float64)

    # Data collection arrays for matplotlib
    num_states_list = []
    times_init = []
    times_add = []
    times_mult = []
    times_norm = []
    times_dynamic = []

    for l in l_values:
        # ==========================================
        # 1. GENERATE MASSIVE 3D RAW DATA
        # ==========================================
        grid_1d = np.arange(-l, l + 1)
        X, Y, Z = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')

        # Flatten into (N, 3) matrix
        raw_states = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)
        num_states = len(raw_states)
        num_states_list.append(num_states)

        # Initial fields pointing perfectly UP in Z: [0.0, 0.0, 1.0]
        raw_fields = np.tile(np.array([0.0, 0.0, 1.0]), (num_states, 1)).astype(np.float64)

        print(f"\nEvaluating l={l} (States: {num_states:,})...")

        # ==========================================
        # 2. INIT & BAKE (create_from_raw)
        # ==========================================
        # Provide extra capacity for our dynamic registration test later
        contract = FieldKernelDataContract(
            state_dimensions=3,
            field_dimensions=3,
            initial_capacity=num_states + 500_000,
            algebra=algebra,
            state_class_ref=State
        )
        storage = NumbaFieldKernelStorage(contract)
        translator = NumbaFieldTranslator()
        utility = NumbaKernelFieldUtility()

        t0 = time.perf_counter()
        manager = FieldComponentManager.create_from_raw(
            contract=contract,
            storage=storage,
            translator=translator,
            utility=utility,
            states=raw_states,
            fields=raw_fields
        )
        t_init = (time.perf_counter() - t0) * 1000
        times_init.append(t_init)

        # ==========================================
        # WARMUP (JIT Compilation for first loop only)
        # ==========================================
        if l == l_values[0]:
            warmup_f = np.tile(np.array([1.0, 0.0, 0.0]), (num_states, 1))
            manager.add_fields(states=raw_states, fields=warmup_f)
            manager.commit_frame()

            # Create a dummy fast ref to warmup utility math
            utility.multiply_mappers_inplace(manager.fast_refs, manager.fast_refs)
            utility.norm_mapper_inplace(manager.fast_refs)

            # Re-initialize the manager to wipe the warmup mutations
            manager = FieldComponentManager.create_from_raw(
                contract, storage, translator, utility, raw_states, raw_fields
            )

        # ==========================================
        # 3. OPERATION: PIPELINE ADDITION
        # ==========================================
        # Adding a cross-wind to every single particle
        cross_wind = np.tile(np.array([1.0, 0.0, 0.0]), (num_states, 1))

        t1 = time.perf_counter()
        manager.add_fields(states=raw_states, fields=cross_wind)
        manager.commit_frame()
        t_add = (time.perf_counter() - t1) * 1000
        times_add.append(t_add)

        # ==========================================
        # 4. OPERATION: RAW SILICON MULTIPLY (In-Place)
        # ==========================================
        # We hit the utility directly since MULTIPLY isn't in FieldCommandType yet
        t2 = time.perf_counter()
        manager.utility.multiply_mappers_inplace(manager.fast_refs, manager.fast_refs)
        t_mult = (time.perf_counter() - t2) * 1000
        times_mult.append(t_mult)

        # ==========================================
        # 5. OPERATION: RAW SILICON NORM (In-Place)
        # ==========================================
        t3 = time.perf_counter()
        manager.utility.norm_mapper_inplace(manager.fast_refs)
        t_norm = (time.perf_counter() - t3) * 1000
        times_norm.append(t_norm)

        # ==========================================
        # 6. OPERATION: DYNAMIC STATE REGISTRATION
        # ==========================================
        # Introduce a brand new grid of 1,000 unmapped states
        new_states = np.random.rand(1000, 3) + 999.0  # Shifted far outside the l-grid
        new_fields = np.ones((1000, 3))

        t4 = time.perf_counter()
        manager.set_fields(states=new_states, fields=new_fields)
        manager.commit_frame()
        t_dynamic = (time.perf_counter() - t4) * 1000
        times_dynamic.append(t_dynamic)

        print(f"  Init & Bake : {t_init:>8.2f} ms")
        print(f"  Pipeline ADD: {t_add:>8.2f} ms")
        print(f"  Raw Multiply: {t_mult:>8.2f} ms")
        print(f"  Raw Norm    : {t_norm:>8.2f} ms")
        print(f"  Dynamic Reg : {t_dynamic:>8.2f} ms")

    # ==========================================
    # 7. GENERATE AND SAVE PLOT
    # ==========================================
    print(f"\nGenerating plot at: {plot_path}")

    plt.figure(figsize=(10, 6))

    # We use a log-log plot because N scales cubically (from 27 to 8,000,000)
    plt.plot(num_states_list, times_init, marker='o', linewidth=2, label='Init & Bake (create_from_raw)')
    plt.plot(num_states_list, times_add, marker='s', linewidth=2, label='Pipeline ADD (Manager Queue)')
    plt.plot(num_states_list, times_mult, marker='^', linewidth=2, label='Raw Multiply (Utility)')
    plt.plot(num_states_list, times_norm, marker='x', linewidth=2, label='Raw Norm (Utility)')
    plt.plot(num_states_list, times_dynamic, marker='d', linewidth=2, linestyle='--',
             label='Dynamic Reg (1k New States)')

    plt.xscale('log')
    plt.yscale('log')

    plt.title('3D Field Component Manager Performance (Hardware Level)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Spatial States (Log Scale)', fontsize=12)
    plt.ylabel('Execution Time in ms (Log Scale)', fontsize=12)

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()

    plt.savefig(plot_path, dpi=300)
    print("Benchmark complete! Check your plot folder.")


if __name__ == "__main__":
    run_3d_benchmark()