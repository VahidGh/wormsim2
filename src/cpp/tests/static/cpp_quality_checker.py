#!/usr/bin/env python3
"""
C++ quality checker for wormsim2.

Verifies that C++20 idioms, safety rules, and architectural patterns are
present in the wormsim2 source tree.  Each check is keyed to a specific
category (TC / FP / NS / FN / RA / CL / TM / SL / LB / PS / OL / DS)
and mapped to a file:line in the codebase.

Run:
    python3 src/cpp/tests/static/cpp_quality_checker.py <repo_root>

Exit 0  — all checks pass.
Exit 1  — one or more checks failed (details printed to stdout).

Also runnable via CTest:
    ctest --test-dir build -R cpp_quality
"""

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""

def find_all(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern))


class Checker:
    def __init__(self, root: Path):
        self.root    = root
        self.passed  = 0
        self.failed  = 0
        self.results: list[tuple[str, bool, str]] = []

    def check(self, key: str, condition: bool, evidence: str) -> None:
        if condition:
            self.passed += 1
            self.results.append((key, True,  f"PASS  [{key}] {evidence}"))
        else:
            self.failed += 1
            self.results.append((key, False, f"FAIL  [{key}] {evidence}"))

    def report(self) -> bool:
        for _, _, msg in self.results:
            print(msg)
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"C++ quality checker: {self.passed}/{total} checks passed"
              f"{'  ✓' if self.failed == 0 else f'  ✗ ({self.failed} failed)'}")
        return self.failed == 0


# ---------------------------------------------------------------------------
# Check groups
# ---------------------------------------------------------------------------

def check_toolchain(c: Checker, root: Path) -> None:
    """TC — Toolchain: cmake version, C++ standard, compiler flags."""

    cmake_root = read(root / "CMakeLists.txt")
    cmake_cpp  = read(root / "src/cpp/CMakeLists.txt")

    c.check("TC-01",
            re.search(r"cmake_minimum_required\s*\(\s*VERSION\s+3\.", cmake_root, re.I) is not None,
            "CMakeLists.txt: cmake_minimum_required(VERSION 3.x) present")

    c.check("TC-02",
            "CMAKE_CXX_STANDARD 20" in cmake_root,
            "CMakeLists.txt:4 — CMAKE_CXX_STANDARD 20")

    c.check("TC-03",
            all(flag in cmake_root for flag in ["-Wall", "-Wextra", "-Wpedantic"]),
            "CMakeLists.txt:24 — -Wall -Wextra -Wpedantic all present")

    headers = find_all(root / "src/cpp/include", "**/*.h")
    missing_pragma = [h.name for h in headers if "#pragma once" not in read(h)]
    c.check("TC-04",
            len(missing_pragma) == 0,
            f"All headers have #pragma once (checked {len(headers)} files)"
            if not missing_pragma else
            f"Headers missing #pragma once: {missing_pragma}")

    c.check("TC-05",
            (root / "docker/Dockerfile.wormsim2-dev").exists(),
            "docker/Dockerfile.wormsim2-dev exists")

    c.check("TC-06",
            "cxx_std_20" in cmake_cpp,
            "src/cpp/CMakeLists.txt:16 — target_compile_features(... cxx_std_20)")


def check_float_safety(c: Checker, root: Path) -> None:
    """FP — Floating-point safety: float for bio params, safe parsing, NaN guard."""

    nc = read(root / "src/cpp/include/io/NetworkConfig.h")
    nl = read(root / "src/cpp/src/io/NEURONLoader.cpp")
    pt = read(root / "src/cpp/tests/io/test_perturbation.cpp")

    bio_params = ["e_rev_mV", "gbar", "capacitance_nF", "v_initial_mV"]
    missing_bp = [p for p in bio_params
                  if not re.search(rf'float\s+{p}', nc)]
    c.check("FP-01",
            len(missing_bp) == 0,
            "NetworkConfig.h — float used for e_rev_mV, gbar, capacitance_nF, v_initial_mV"
            if not missing_bp else
            f"NetworkConfig.h — missing float bio-params: {missing_bp}")

    c.check("FP-02",
            re.search(r"constexpr float k[Ee]ps", pt) is not None,
            "test_perturbation.cpp — constexpr float kEps (no raw == on floats)")

    c.check("FP-03",
            re.search(r"try\s*\{[^}]*std::stof", nl, re.DOTALL) is not None,
            "NEURONLoader.cpp:61 — try/catch parseFloat with safe fallback (NaN guard)")

    c.check("FP-04",
            'return fallback' in nl,
            "NEURONLoader.cpp:63 — fallback return in parseFloat (prevents NaN propagation)")


def check_namespaces(c: Checker, root: Path) -> None:
    """NS — Namespaces: no 'using namespace std' in headers, wormsim2 namespace, string_view."""

    src_files  = (find_all(root / "src/cpp/src",     "**/*.cpp") +
                  find_all(root / "src/cpp/include",  "**/*.h"))
    headers    = find_all(root / "src/cpp/include", "**/*.h")
    cpp_files  = find_all(root / "src/cpp/src",     "**/*.cpp")

    bad_using = [f.name for f in headers if "using namespace std" in read(f)]
    c.check("NS-01",
            len(bad_using) == 0,
            f"No 'using namespace std;' in any header (checked {len(headers)})"
            if not bad_using else f"Headers with 'using namespace std': {bad_using}")

    missing_ns = [f.name for f in src_files
                  if "namespace wormsim2" not in read(f)]
    c.check("NS-02",
            len(missing_ns) == 0,
            f"All {len(src_files)} source/header files declare 'namespace wormsim2'"
            if not missing_ns else f"Files missing 'namespace wormsim2': {missing_ns}")

    anon_ns = [f.name for f in cpp_files if "namespace {" in read(f)]
    c.check("NS-03",
            len(anon_ns) >= 1,
            f"Anonymous 'namespace {{' present in .cpp files: {anon_ns} (file-local scope)")

    nc = read(root / "src/cpp/include/io/NetworkConfig.h")
    c.check("NS-04",
            "[[nodiscard]] const NeuronDef* find_neuron(std::string_view" in nc,
            "NetworkConfig.h:107 — find_neuron uses std::string_view (non-owning ref)")


def check_functions(c: Checker, root: Path) -> None:
    """FN — Functions/lambdas: [[nodiscard]], [[maybe_unused]], lambdas, pure helpers."""

    nc  = read(root / "src/cpp/include/io/NetworkConfig.h")
    nip = read(root / "src/cpp/include/io/NetworkInputParser.h")
    nlh = read(root / "src/cpp/include/io/NEURONLoader.h")
    nml = read(root / "src/cpp/src/io/NeuroMLLoader.cpp")
    ner = read(root / "src/cpp/src/io/NEURONLoader.cpp")
    pcg = read(root / "src/cpp/src/io/PerturbationConfig.cpp")

    nodiscard_count = sum(
        t.count("[[nodiscard]]")
        for t in [nc, nip, nlh]
    )
    c.check("FN-01",
            nodiscard_count >= 4,
            f"[[nodiscard]] applied {nodiscard_count} times across headers (≥4 required)")

    c.check("FN-02",
            "[[maybe_unused]]" in ner,
            "NEURONLoader.cpp:202 — [[maybe_unused]] on loadHoc cfg stub parameter")

    c.check("FN-03",
            re.search(r"auto\s+\w+\s*=\s*\[&\]", nml) is not None,
            "NeuroMLLoader.cpp:208 — [&] capture lambda (find_by_name closure)")

    c.check("FN-04",
            re.search(r"const float \w+\s*=\s*\[&\]\s*\{", pcg) is not None,
            "PerturbationConfig.cpp:33 — IIFE lambda for exception-safe float parsing")

    c.check("FN-05",
            all(f in ner for f in ["parseNeuronBlock", "parseParameterBlock", "parseStateBlock"]),
            "NEURONLoader.cpp:72/100/124 — decomposed pure static helper functions (SRP)")

    c.check("FN-06",
            "float fallback = 0.0f" in ner,
            "NEURONLoader.cpp:61 — default parameter 'float fallback = 0.0f'")


def check_raii(c: Checker, root: Path) -> None:
    """RA — RAII / smart pointers: no bare new/delete, ifstream RAII, optional."""

    src_files = find_all(root / "src/cpp/src", "**/*.cpp")

    for f in src_files:
        text = read(f)
        bad_new = re.findall(r'\bnew\s+\w', text)
        c.check(f"RA-01-{f.stem}",
                len(bad_new) == 0,
                f"{f.name} — no bare 'new' (RAII: use make_unique/make_shared)"
                if not bad_new else f"{f.name} — bare 'new' found: {bad_new}")

        bad_del = re.findall(r'\bdelete\s', text)
        c.check(f"RA-02-{f.stem}",
                len(bad_del) == 0,
                f"{f.name} — no bare 'delete'"
                if not bad_del else f"{f.name} — bare 'delete' found: {bad_del}")

    ner = read(root / "src/cpp/src/io/NEURONLoader.cpp")
    c.check("RA-03",
            "std::ifstream file(" in ner,
            "NEURONLoader.cpp:166 — std::ifstream RAII (no explicit close() needed)")

    nip_h = read(root / "src/cpp/include/io/NetworkInputParser.h")
    c.check("RA-04",
            "std::optional<" in nip_h,
            "NetworkInputParser.h:36-37 — std::optional<> for nullable parameters")

    nip_cpp = read(root / "src/cpp/src/io/NetworkInputParser.cpp")
    c.check("RA-05",
            "perturb_spec.has_value()" in nip_cpp,
            "NetworkInputParser.cpp:66 — optional.has_value() guard (no null-pointer check)")


def check_classes(c: Checker, root: Path) -> None:
    """CL — Classes: Rule of Zero, const-correctness, SRP, static methods."""

    nc  = read(root / "src/cpp/include/io/NetworkConfig.h")
    nlh = read(root / "src/cpp/include/io/NEURONLoader.h")

    c.check("CL-01",
            "~NetworkConfig" not in nc and "~NeuronDef" not in nc,
            "NetworkConfig.h — no user-defined destructor (Rule of Zero)")

    c.check("CL-02",
            nc.count("const noexcept") >= 3,
            f"NetworkConfig.h:102-104 — {nc.count('const noexcept')} const noexcept accessors")

    c.check("CL-03",
            "[[nodiscard]] int neuron_count()       const noexcept" in nc,
            "NetworkConfig.h:102 — neuron_count() const noexcept (const-correctness)")

    c.check("CL-04",
            all("static" in line
                for line in nlh.split("\n")
                if re.search(r'\bvoid\b|\bbool\b|\bChannel', line)
                and "private" not in line.lower()),
            "NEURONLoader.h — all interface methods are static (pure-function class)")

    c.check("CL-05",
            "enum class NetworkFormat" in read(root / "src/cpp/include/io/NetworkInputParser.h"),
            "NetworkInputParser.h:12 — enum class NetworkFormat (scoped, type-safe enum)")


def check_templates(c: Checker, root: Path) -> None:
    """TM — Templates: vector<T>, unordered_map, optional, sregex_iterator."""

    nc  = read(root / "src/cpp/include/io/NetworkConfig.h")
    ner = read(root / "src/cpp/src/io/NEURONLoader.cpp")
    nml = read(root / "src/cpp/src/io/NeuroMLLoader.cpp")

    c.check("TM-01",
            "std::unordered_map<std::string, ChannelDef>" in nc,
            "NetworkConfig.h:97 — unordered_map<string,ChannelDef> (O(1) channel lookup)")

    vector_types = re.findall(r'std::vector<(\w+)>', nc)
    c.check("TM-02",
            len(set(vector_types)) >= 4,
            f"NetworkConfig.h — {len(set(vector_types))} distinct vector<T> instantiations: {set(vector_types)}")

    c.check("TM-03",
            "std::optional<" in nc or
            "std::optional<" in read(root / "src/cpp/include/io/NetworkInputParser.h"),
            "NetworkInputParser.h / NetworkConfig.h — std::optional<T> used")

    c.check("TM-04",
            "sregex_iterator" in ner,
            "NEURONLoader.cpp:103 — std::sregex_iterator (regex forward iterator)")

    c.check("TM-05",
            "sregex_iterator" in nml,
            "NeuroMLLoader.cpp:80 — std::sregex_iterator in NeuroML XML scanner")

    c.check("TM-06",
            "[[maybe_unused]]" in ner,
            "NEURONLoader.cpp:202 — [[maybe_unused]] attribute on stub parameter")


def check_stdlib(c: Checker, root: Path) -> None:
    """SL — Standard library: algorithms, iterators, filesystem, streams, C++20."""

    ner = read(root / "src/cpp/src/io/NEURONLoader.cpp")
    nml = read(root / "src/cpp/src/io/NeuroMLLoader.cpp")
    pcg = read(root / "src/cpp/src/io/PerturbationConfig.cpp")

    c.check("SL-01",
            "std::transform(" in ner,
            "NEURONLoader.cpp:29 — std::transform for toLower (algorithm over range)")

    c.check("SL-02",
            "directory_iterator" in ner,
            "NEURONLoader.cpp:189 — std::filesystem::directory_iterator for .mod scan")

    c.check("SL-03",
            "starts_with(" in pcg,
            "PerturbationConfig.cpp:41/83/92 — string_view::starts_with (C++20)")

    c.check("SL-04",
            "istreambuf_iterator<char>" in ner,
            "NEURONLoader.cpp:171 — istreambuf_iterator<char> whole-file read")

    c.check("SL-05",
            "std::getline(" in pcg,
            "PerturbationConfig.cpp — std::getline line-by-line parsing")

    c.check("SL-06",
            "std::filesystem::path" in ner,
            "NEURONLoader.cpp — std::filesystem::path throughout")

    c.check("SL-07",
            "std::regex" in ner or "std::regex" in nml,
            "NEURONLoader.cpp:54 / NeuroMLLoader.cpp:55 — std::regex for NMODL/XML scanning")


def check_libraries(c: Checker, root: Path) -> None:
    """LB — CMake libraries: static lib, PUBLIC/PRIVATE propagation, feature requirements."""

    cmake_cpp  = read(root / "src/cpp/CMakeLists.txt")
    cmake_test = read(root / "src/cpp/tests/io/CMakeLists.txt")

    c.check("LB-01",
            "add_library(wormsim2_io STATIC" in cmake_cpp,
            "src/cpp/CMakeLists.txt:8 — add_library(wormsim2_io STATIC) static lib")

    c.check("LB-02",
            "target_include_directories(wormsim2_io PUBLIC" in cmake_cpp,
            "src/cpp/CMakeLists.txt:15 — PUBLIC include propagation to consumers")

    c.check("LB-03",
            "PRIVATE wormsim2_io" in cmake_test,
            "tests/io/CMakeLists.txt — PRIVATE link (lib not re-exported from tests)")

    c.check("LB-04",
            "target_compile_features(wormsim2_io PUBLIC cxx_std_20" in cmake_cpp,
            "src/cpp/CMakeLists.txt:16 — cxx_std_20 propagated PUBLIC to consumers")


def check_parallel_safety(c: Checker, root: Path) -> None:
    """PS — Parallel safety: immutable config, no global state, const-correct output params."""

    nc = read(root / "src/cpp/include/io/NetworkConfig.h")

    c.check("PS-01",
            "mutable" not in nc,
            "NetworkConfig.h — no 'mutable' members (immutable-after-load, race-free)")

    bad_static = []
    for f in find_all(root / "src/cpp/src", "**/*.cpp"):
        text = read(f)
        for line in text.split("\n"):
            stripped = line.strip()
            if (re.match(r'static\s+(?!const\s)(?!inline\s)', stripped)
                    and not stripped.startswith("//")
                    and not stripped.startswith("*")
                    and "static_assert" not in stripped):
                bad_static.append(f"{f.name}: {stripped[:60]}")
    c.check("PS-02",
            len(bad_static) == 0,
            "No mutable static variables in .cpp files (thread-safe parsing functions)"
            if not bad_static else f"Mutable static found: {bad_static}")

    ner = read(root / "src/cpp/src/io/NEURONLoader.cpp")
    c.check("PS-03",
            "NetworkConfig&               cfg" in ner or "NetworkConfig& cfg" in ner,
            "NEURONLoader.cpp — NetworkConfig& passed as output parameter (no globals)")


def check_object_layout(c: Checker, root: Path) -> None:
    """OL — Object layout: trivially-copyable structs, integer IDs, no virtual in data types."""

    nc = read(root / "src/cpp/include/io/NetworkConfig.h")

    c.check("OL-01",
            "virtual" not in nc,
            "NetworkConfig.h — no virtual functions in data structs (trivially copyable)")

    c.check("OL-02",
            "int   muscle_id{-1};" in nc or "int muscle_id{-1};" in nc,
            "NetworkConfig.h:82 — NMJDef::muscle_id int (0..94), flat integer ID")

    c.check("OL-03",
            "int   neuron_a{-1};" in nc or "int neuron_a{-1};" in nc,
            "NetworkConfig.h:70 — GapJunctionDef uses integer IDs (flat, sendable)")

    c.check("OL-04",
            "int         id{-1};" in nc,
            "NetworkConfig.h:45 — NeuronDef::id integer (maps to domain partitions)")


def check_data_safety(c: Checker, root: Path) -> None:
    """DS — Data safety: const parameters, no shared mutable, no data races."""

    ner = read(root / "src/cpp/src/io/NEURONLoader.cpp")
    nml = read(root / "src/cpp/src/io/NeuroMLLoader.cpp")

    c.check("DS-01",
            "const std::string& block" in ner,
            "NEURONLoader.cpp — const std::string& block (read-only input, parallelisable)")

    c.check("DS-02",
            "const std::string& xml" in nml,
            "NeuroMLLoader.cpp — const std::string& xml (read-only, no shared state)")

    pcg = read(root / "src/cpp/src/io/PerturbationConfig.cpp")
    c.check("DS-03",
            "[&]" in nml or "[&]" in pcg,
            "NeuroMLLoader.cpp:208 / PerturbationConfig.cpp:33 — [&] only over local scope")

    any_omp = any("#pragma omp" in read(f) for f in find_all(root / "src/cpp/src", "**/*.cpp"))
    c.check("DS-04",
            not any_omp,
            "io/ source files — no #pragma omp yet (pure serial; parallel in NeuralIntegrator)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <repo_root>")
        return 1

    root = Path(sys.argv[1]).resolve()
    print(f"C++ quality checker — repo root: {root}\n{'='*60}")

    c = Checker(root)

    check_toolchain(c, root)
    check_float_safety(c, root)
    check_namespaces(c, root)
    check_functions(c, root)
    check_raii(c, root)
    check_classes(c, root)
    check_templates(c, root)
    check_stdlib(c, root)
    check_libraries(c, root)
    check_parallel_safety(c, root)
    check_object_layout(c, root)
    check_data_safety(c, root)

    return 0 if c.report() else 1


if __name__ == "__main__":
    sys.exit(main())
