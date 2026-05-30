# Contributing to PyRobotics

Thank you for your interest in contributing! This document provides guidelines for contributions.

## Contribution Process

1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests to verify (`python tests/test_all_modules.py`)
5. **Commit** with a descriptive message
6. Submit a **Pull Request**

## Code Style

This project follows a strict scientific computing coding convention:

- **Pure functions + array data flow**: No classes (except `LatestResult` and `_GridWorld`)
- **Chain suffix notation**: `_raw → _u → _det → _smooth → _cal` for processing states
- **Phase organization**: Code sections marked with `# === Phase N: Description ===`
- **Module decoupling**: Inter-module communication via Protobuf messages only
- **Vectorized operations**: No `for` loops over array elements (convergence outer loops excepted)
- **Physical-unit parameters**: Window sizes, thresholds in physical units (cm⁻¹, µm, °)
- **No docstrings/comments**: Only Phase structural headers; function signatures are self-documenting
- **No try/except**: Use `assert` → `if + raise ValueError` for pre-condition validation
- **No hardcoded parameters**: All algorithm parameters exposed as function arguments with defaults

## Commit Messages

Format: `type(scope): description`

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

Examples:
- `feat(tracking): add MPC terminal cost`
- `fix(localization): EKF Joseph-form covariance update`
- `test(slam): add FastSLAM adaptive threshold test`

## Testing

All changes must pass the existing test suite:

```bash
python tests/test_all_modules.py
```

New features should include corresponding test cases.

## Protobuf Changes

If modifying `.proto` files, regenerate the Python code:

```bash
python -m grpc_tools.protoc -I=proto --python_out=generated proto/*.proto
```
