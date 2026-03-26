# Python 打包断层期问题：pkg_resources 报错解析

## 问题背景

在安装 `flatdict==4.0.1` 时遇到 `ModuleNotFoundError: No module named 'pkg_resources'` 错误，这是一个典型的**Python 打包技术新旧更替断层期**问题。

简单来说，`flatdict==4.0.1` 是一个"老古董"（2020年发布），而现代工具（Python 3.11 和 `uv`）与之存在严重的**代沟**。

## 三个核心原因

### 1. 消失的"胶水"：`pkg_resources` 的弃用

在老派 Python 开发中，`setuptools` 附带了一个叫 `pkg_resources` 的工具，几乎所有老项目都用它来管理版本和元数据。

- **现状：** 现代 Python 社区（尤其是 `setuptools` v70 之后）认为这个工具太臃肿、性能差，于是把它**彻底删除了**。
- **冲突：** `flatdict 4.0.1` 的安装脚本里写死了 `import pkg_resources`。当用最新工具安装时，程序会因为找不到这个"胶水"而崩溃。

### 2. `uv` 的"纯净实验室"机制（Build Isolation）

为了保证安全和一致性，现代工具如 `uv` 或 `pip` 在安装包时，会先创建一个**临时、干净的虚拟环境**（类似于沙盒）来编译这个包。

- **问题：** 这个沙盒默认只装最先进的工具。
- **后果：** 老项目在沙盒里既找不到 `pkg_resources`，也找不到用来打包的 `wheel`。这就是为什么即使在外面装了 `setuptools`，报错却依然存在——沙盒里还是没有。

### 3. "手动挡"与"自动挡"的矛盾

`flatdict 4.0.1` 属于"手动挡"时代，它假设系统里已经预装好了 `setuptools`、`wheel` 等所有零配件。而 `uv` 是追求极简和速度的"自动挡"，它不希望环境里堆满这些构建工具。

使用 `--no-build-isolation` 参数本质上是**强行关掉了沙盒**，告诉 `uv`："别去开那个临时环境了，直接用现有的这些老零件来修这个老古董。"

## 为什么机器人/强化学习项目经常遇到这种问题？

`lehome-challenge` 涉及 **Isaac Lab**，这类机器人仿真项目通常依赖极其复杂的底层 C++ 库和特定的 Python 依赖链：

- 很多科研代码或早期框架为了稳定，会**锁死（Pin）**依赖版本
- 这导致了明明用着最新的 Ubuntu 22.04/24.04 和 Python 3.11，却不得不为了兼容某个几年前的算法包，回头去手动修补那些早已过时的打包工具

## 快速识别与解决

遇到类似 `ModuleNotFoundError: No module named 'pkg_resources'` 的报错，基本都可以断定是**"新环境装老包"**的问题。

### 解决方案

**方案 1：添加 extra-build-dependencies（推荐）**

在 `pyproject.toml` 中添加：

```toml
[tool.uv.extra-build-dependencies]
flatdict = ["setuptools"]
```

**方案 2：禁用构建隔离**

```bash
uv pip install setuptools
uv pip install flatdict==4.0.1 --no-build-isolation
```

**方案 3：使用旧版 setuptools**

如果上述方案都不行，可能需要降级 setuptools：

```bash
uv pip install "setuptools<70"
```

## 相关链接

- [Setuptools v70 Release Notes](https://setuptools.pypa.io/en/stable/history.html)
- [uv Build Isolation Documentation](https://docs.astral.sh/uv/pip/build-isolation/)
- [flatdict on PyPI](https://pypi.org/project/flatdict/)
