# Contributing to GhostFlow

Thank you for your interest in contributing to GhostFlow! We welcome contributions from everyone.

## 🎯 Ways to Contribute

- 🐛 **Report Bugs** - Found a bug? Open an issue!
- 💡 **Suggest Features** - Have an idea? We'd love to hear it!
- 📝 **Improve Documentation** - Help make our docs better
- 🔧 **Submit Code** - Fix bugs or implement features
- 🧪 **Write Tests** - Help us improve test coverage
- ⚡ **Optimize Performance** - Make GhostFlow faster!

## 🚀 Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/choksi2212/ghost-flow.git
cd ghost-flow
```

### 2. Build the Project

```bash
cargo build --workspace
```

### 3. Run Tests

```bash
cargo test --workspace
```

### 4. Make Your Changes

Create a new branch for your feature:

```bash
git checkout -b feature/your-feature-name
```

## 📋 Development Guidelines

### Code Style

- Follow Rust standard formatting: `cargo fmt`
- Run clippy: `cargo clippy --workspace`
- Ensure zero warnings: `cargo build --workspace 2>&1 | grep warning`

### Testing

- Add tests for new features
- Ensure all existing tests pass
- Aim for high test coverage

### Documentation

- Document public APIs with doc comments
- Include examples in doc comments
- Update README.md if needed

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add support for LSTM layers
fix: Correct gradient computation in Conv2d
docs: Update installation instructions
perf: Optimize matrix multiplication with SIMD
```

## 🔍 Code Review Process

1. Submit a pull request
2. Automated tests will run
3. Maintainers will review your code
4. Address any feedback
5. Once approved, your PR will be merged!

## 🎨 Module Structure

GhostFlow is organized into modules:

- `ghostflow-core` - Core tensor operations
- `ghostflow-nn` - Neural network layers
- `ghostflow-optim` - Optimizers
- `ghostflow-data` - Data loading
- `ghostflow-autograd` - Automatic differentiation
- `ghostflow-ml` - Machine learning algorithms
- `ghostflow-cuda` - GPU acceleration

## 🐛 Reporting Bugs

When reporting bugs, please include:

- Rust version (`rustc --version`)
- GhostFlow version
- Minimal code to reproduce the issue
- Expected vs actual behavior
- Error messages or stack traces

## 💡 Suggesting Features

When suggesting features:

- Explain the use case
- Describe the proposed API
- Consider performance implications
- Check if it aligns with GhostFlow's goals

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as GhostFlow (MIT/Apache-2.0).

## 🙏 Thank You!

Every contribution, no matter how small, helps make GhostFlow better. We appreciate your time and effort!

---

**Questions?** Open an issue or join our Discord community!
