"""
miv_neuroh5_io_test.py — Parallel neuroh5 I/O tests against Microcircuit data
Run via:  mpiexec -n N python3 miv_neuroh5_io_test.py

Tests:
  1. scatter_read_trees on PYR_forest.h5, OLM_forest.h5, PVBC_forest.h5
  2. scatter_read_cell_attributes on MiV_Cells_Microcircuit_Small_20220410.h5
  3. scatter_read_graph on MiV_Connections_Microcircuit_Small_20220410.h5
  4. read_cell_attributes on MiV_input_features.h5 / MiV_input_spikes.h5
"""

import sys
import time
import traceback

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

DATA = "/lus/grand/projects/gpu_hack/iowarp/neuroh5"
SMALL = f"{DATA}/Microcircuit_Small"
RESULTS = []


def section(name):
    if rank == 0:
        print(f"\n{'='*60}", flush=True)
        print(f"  {name}", flush=True)
        print(f"{'='*60}", flush=True)


def ok(msg):
    if rank == 0:
        print(f"  [OK] {msg}", flush=True)


def fail(msg):
    if rank == 0:
        print(f"  [FAIL] {msg}", flush=True)


def run_test(label, fn):
    comm.Barrier()
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        ok(f"{label} ({elapsed:.2f}s) → {result}")
        RESULTS.append((label, "PASS", result, elapsed))
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        fail(f"{label}: {e}")
        if rank == 0:
            print(tb[:1000], flush=True)
        RESULTS.append((label, "FAIL", str(e)[:120], elapsed))
    comm.Barrier()


from neuroh5.io import (
    scatter_read_trees,
    scatter_read_cell_attributes,
    scatter_read_graph,
    read_population_ranges,
    read_population_names,
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: scatter_read_trees
# ─────────────────────────────────────────────────────────────────────────────
section("Test 1: scatter_read_trees")


def test_forest(pop, fname):
    path = f"{DATA}/{fname}"
    # io_size=1 — one rank does all I/O then distributes; avoids HDF5 parallel
    # hyperslab partitioning bugs with 1.14.x GNU variant
    gen, _ = scatter_read_trees(path, pop, io_size=1, comm=comm)
    count = 0
    total_sections = 0
    for gid, tree in gen:
        count += 1
        total_sections += int(np.sum(tree.get("section", [0])))
    count_all = comm.reduce(count, op=MPI.SUM, root=0)
    secs_all = comm.reduce(total_sections, op=MPI.SUM, root=0)
    if rank == 0:
        return f"{pop}: {count_all} cells, {secs_all} section entries"
    return None


# PYR_forest.h5: Attribute Pointer has 0 elements (incomplete export) — skip
# PYR_forest_compressed.h5: complete (80001 elements) — use this instead
# Run small forests first to check if scatter works at all on Polaris
run_test("scatter_read_trees OLM_forest.h5",            lambda: test_forest("OLM",  "OLM_forest.h5"))
run_test("scatter_read_trees PVBC_forest.h5",           lambda: test_forest("PVBC", "PVBC_forest.h5"))
# PYR_forest_compressed.h5: 80k cells (~4.4 GB) — large alltoallv triggers CMA
run_test("scatter_read_trees PYR_forest_compressed.h5", lambda: test_forest("PYR", "PYR_forest_compressed.h5"))


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: scatter_read_cell_attributes (Generated Coordinates)
# ─────────────────────────────────────────────────────────────────────────────
section("Test 2: scatter_read_cell_attributes (Microcircuit_Small coordinates)")


def test_cell_coords(pop):
    path = f"{SMALL}/MiV_Cells_Microcircuit_Small_20220410.h5"
    result = scatter_read_cell_attributes(
        path, pop, namespaces=["Generated Coordinates"],
        comm=comm, io_size=1
    )
    # API returns (generator, pop_ranges) or just generator depending on version
    gen = result[0] if isinstance(result, (tuple, list)) else result
    gids = [item[0] if isinstance(item, (tuple, list)) else item for item in gen]
    count = comm.reduce(len(gids), op=MPI.SUM, root=0)
    if rank == 0:
        return f"{pop}: {count} cells"
    return None


for pop in ["PYR", "OLM", "PVBC", "STIM"]:
    run_test(f"scatter_read_cell_attributes {pop}",
             lambda p=pop: test_cell_coords(p))


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: scatter_read_graph (Microcircuit_Small connections)
# ─────────────────────────────────────────────────────────────────────────────
section("Test 3: scatter_read_graph (MiV_Connections_Microcircuit_Small_20220410.h5)")


def test_connections():
    path = f"{SMALL}/MiV_Connections_Microcircuit_Small_20220410.h5"
    result = scatter_read_graph(path, comm=comm, io_size=1)
    # API: returns (graph_dict, pop_ranges) tuple or a single dict
    if isinstance(result, tuple):
        graph = result[0]
    else:
        graph = result
    total_edges = 0
    projections = []
    for key, edges_dict in graph.items():
        if isinstance(edges_dict, dict):
            for gid, adj in edges_dict.items():
                if isinstance(adj, tuple) and len(adj) >= 1:
                    src_ids = adj[0]
                    total_edges += len(src_ids) if hasattr(src_ids, '__len__') else 0
            if edges_dict:
                projections.append(f"{key}:{len(edges_dict)}")
    total_all = comm.reduce(total_edges, op=MPI.SUM, root=0)
    if rank == 0:
        return f"{total_all} edges  [{', '.join(projections[:4])}]"
    return None


run_test("scatter_read_graph connections", test_connections)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: read_population_ranges / names
# ─────────────────────────────────────────────────────────────────────────────
section("Test 4: population metadata")


def test_pop_names(path, label):
    names = read_population_names(path)
    ranges, total = read_population_ranges(path)
    if rank == 0:
        return f"{label}: {names}, total={total}"
    return None


run_test("pop names MiV_Cells_Small",
         lambda: test_pop_names(f"{SMALL}/MiV_Cells_Microcircuit_Small_20220410.h5", "cells"))
run_test("pop names MiV_Connections_Small",
         lambda: test_pop_names(f"{SMALL}/MiV_Connections_Microcircuit_Small_20220410.h5", "conns"))


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: OLM/PVBC synapses (forest_syns files)
# ─────────────────────────────────────────────────────────────────────────────
section("Test 5: scatter_read_cell_attributes (Synapse Attributes)")


def test_synapses(pop, fname):
    path = f"{DATA}/{fname}"
    result = scatter_read_cell_attributes(
        path, pop, namespaces=["Synapse Attributes"],
        comm=comm, io_size=1
    )
    gen = result[0] if isinstance(result, (tuple, list)) else result
    count = sum(1 for _item in gen)
    count_all = comm.reduce(count, op=MPI.SUM, root=0)
    if rank == 0:
        return f"{pop}: {count_all} cells with synapse attrs"
    return None


run_test("scatter_read_cell_attrs OLM_forest_syns",
         lambda: test_synapses("OLM", "OLM_forest_syns.h5"))
run_test("scatter_read_cell_attrs PVBC_forest_syns",
         lambda: test_synapses("PVBC", "PVBC_forest_syns.h5"))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
if rank == 0:
    section("Summary")
    passed = sum(1 for _, s, _, _ in RESULTS if s == "PASS")
    total = len(RESULTS)
    print(f"\n  {passed}/{total} tests passed\n")
    for label, status, detail, elapsed in RESULTS:
        mark = "PASS" if status == "PASS" else "FAIL"
        print(f"  {mark}  [{elapsed:.2f}s]  {label}")
        if detail and rank == 0:
            print(f"         → {detail}")
    print()
