# wormsim2 — Modern C++ Best-Practice Checklist

> Referenced by NFR-QUAL-01. Applies to all C++ source under `src/cpp/`.
> The checklist is verified by the CI pipeline (cppcheck, clang-tidy) and code review.

---

## 1. Language standard and compiler flags

- Minimum standard: **C++20**. Use C++23 features where the project compiler supports
  them and the feature meaningfully improves clarity or performance; do not artificially
  restrict to an older dialect.
- Prefer newer standard features over workarounds: e.g. `std::string_view::starts_with`
  (C++20), `std::span` (C++20), `std::format` (C++20), `std::ranges` (C++20),
  `std::expected` (C++23) — use them directly rather than reimplementing.
- Build with `-Wall -Wextra -Wpedantic`; all warnings shall be addressed (not suppressed).
- Enable `-fno-exceptions` only where specifically justified by a hot path; the default is
  exceptions enabled.
- Use `-DNDEBUG` for release builds; `assert()` is permitted for internal invariants.

## 2. Resource management (RAII)

- **No raw `new` / `delete`** in user code; use `std::make_unique<T>` and `std::make_shared<T>`.
- **No raw owning pointers** as data members or function return types.
- File handles, sockets, and OS resources: wrap in RAII types or `std::unique_ptr` with a
  custom deleter.
- Prefer stack allocation; heap-allocate only when lifetime or size demands it.

## 3. Const-correctness

- Every method that does not modify `*this` shall be declared `const`.
- Prefer `const T&` parameters over `T` for types that are non-trivially copyable.
- Use `const` for all local variables that are not reassigned.
- Declare `constexpr` functions and variables wherever the value is known at compile time.

## 4. Type safety and modern idioms

- Prefer `std::optional<T>` over nullable raw pointers or sentinel values for optional results.
- Use `std::variant<Ts…>` over tag-unions or `void*`.
- Use structured bindings (`auto [a, b] = …`) for multi-return functions.
- Use `if constexpr` in templates instead of SFINAE; use **concepts** (C++20) to
  constrain template parameters instead of `enable_if`.
- Apply `[[nodiscard]]` to functions whose return value must not be silently discarded.
- Apply `[[maybe_unused]]` rather than casting to `void` for intentionally unused parameters.
- Prefer scoped enumerations (`enum class`) over unscoped `enum`.

## 5. Standard library containers and algorithms

- Default container: `std::vector<T>` (cache-friendly, predictable).
- Use `std::unordered_map<K,V>` for O(1) key lookups; `std::map` only when ordered
  iteration is needed.
- Use standard algorithms (`std::transform`, `std::accumulate`, `std::sort`, …) in
  preference to hand-written loops.
- Reserve vector capacity when the size is known before insertion (`vec.reserve(n)`).
- Avoid iterator invalidation: do not modify a container while iterating over it.

## 6. Performance-critical data layouts

- For arrays accessed in tight loops or dispatched to compute backends, use
  **Structure-of-Arrays (SoA)** layout rather than Array-of-Structures (AoS) to maximise
  SIMD/GPU memory-access efficiency.
- `NetworkConfig` (loaded from files) uses AoS for readability; `NeuralIntegrator` and
  `FEMBody` convert to SoA on startup.
- Avoid `std::string` in hot-path data structs; use integer IDs or `std::string_view`.
- Annotate alignment with `alignas(N)` where SIMD or GPU DMA requires it.

## 7. Error handling

- Use **exceptions** for errors that cross module boundaries (parse errors, I/O errors,
  invalid configuration).
- Use `std::optional<T>` for within-module functions that may legitimately find no result.
- Error messages shall include the source location (`__FILE__`, `__LINE__`) and enough
  context for the caller to diagnose the problem without a debugger.
- Never swallow exceptions silently; catch only what can be handled, rethrow or translate
  the rest.

## 8. Thread safety (for pipeline components)

- Clearly document whether each class is thread-safe, not thread-safe, or safe for
  concurrent reads.
- Data shared between the simulation pipeline and the asynchronous `OutputSerialiser`
  shall be protected by a mutex or written via a lock-free queue.
- Prefer immutable data (loaded once, then const) over shared mutable state.

## 9. Static analysis

- `cppcheck --std=c++20 --enable=all` shall report zero errors and zero warnings on all
  committed source (style issues exempted per project `.cppcheck` suppression file).
- `clang-tidy` with the project `.clang-tidy` config shall produce zero warnings on
  new source files.
- CI enforces both checks on every pull request (FR-DEPLOY-03).

## 10. File and module organisation

- One primary class or closely related group of types per header file.
- Headers use `#pragma once` (no include guards).
- Implementation files include only their own header first, then standard headers, then
  third-party headers — in alphabetical order within each group.
- No `using namespace std;` in headers; permitted (sparingly) in `.cpp` files.
- Forward-declare types in headers where possible; include the full definition only where
  the size is needed.

## 11. Documentation and comments

- Public API functions: one-line Doxygen `/// brief description` is sufficient.
- Comment the *why*, not the *what*: explain non-obvious constraints, workarounds, or
  invariants; do not re-state what the code already says.
- No commented-out code in committed files.

---

*This checklist is enforced by CI (NFR-QUAL-01) and reviewed on every pull request.*
