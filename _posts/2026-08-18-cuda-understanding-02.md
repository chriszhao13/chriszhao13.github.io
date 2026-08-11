---
layout: post
title: 我所理解的 CUDA · 第 1 步：编程模型（Guide §5）
date: 2026-08-18 10:00:00+0800
description: 从最简单的 Kernel 出发，拆解线程层次（grid → block → warp → thread）、内存层次与异步 SIMT 编程模型。
tags: cuda
featured: false
---

> 本节对应《CUDA C++ Programming Guide》Release 13.3 **第 5 章 Programming Model**。这是整个 CUDA 的地基——先建立心智模型，再谈 API 和优化。

## 从一个最简单的 Kernel 说起

```cpp
__global__ void add(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

// 主机端调用
add<<<gridDim, blockDim>>>(d_out, d_a, d_b, n);
```

这一小段代码背后藏着 CUDA 的全部核心抽象。我们逐个拆。

## Kernels：在设备上执行的函数

用 `__global__` 修饰的函数称为 **kernel**：

- 由**主机（host）**调用，在**设备（device）**上执行；
- 调用时通过 `<<<grid, block>>>` 指定执行配置（后面细讲）；
- kernel 启动是**异步**的：主机发出启动请求后不等待完成，立即继续执行。

> 注意：从 CUDA 12.x 起，`<<<>>>` 只是 `cudaLaunchKernel` 的语法糖。了解底层有助于后面理解 CUDA Graphs。

## 线程层次：grid → block → thread

执行 kernel 的线程组织成**两层**结构：

```
grid（网格）           ← 一次 kernel 启动的所有线程
└── block（线程块）     ← 可以由多个线程组成
    └── thread（线程）  ← 最小执行单元
```

- `gridDim`：网格中 block 的数量（`gridDim.x` / `.y` / `.z`）；
- `blockDim`：每个 block 中 thread 的数量（最多 1024 个）；
- `blockIdx`：当前 block 在 grid 中的索引；
- `threadIdx`：当前 thread 在 block 中的索引。

每个线程计算自己的全局索引（上面的经典公式）：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

### 为什么这样分层？

**因为可扩展性（Scalability）**——这是 Guide §3.3 反复强调的：

- 软件上，你只需描述"数据并行"的结构（几个 block、每个 block 几个线程）；
- 硬件上，SM 按需调度 block，**同一份 kernel 代码在 1 个 SM 和 128 个 SM 的 GPU 上都能跑**，不用改代码。

### Thread Block Clusters（较新的抽象）

从 Hopper 架构（Compute Capability 9.0）开始，Guide 引入了**第三层**：线程块集群（Thread Block Clusters）。

```
grid
└── cluster（集群）   ← 多个 block 组成，可访问彼此的共享内存（DSM）
    └── block
        └── thread
```

- 集群内的 block 保证**同时被调度**到同一 GPU 的多个 SM 上（co-resident）；
- 通过**分布式共享内存（Distributed Shared Memory）**，一个 block 可以直接读/写集群内其他 block 的共享内存；
- 编程模型上新增了 `clusterDim`、`clusterIdx` 等内置变量。

> 这是"最新特性"里第一个真正改变线程层次的东西。第 5 步我们会用专门篇幅深入 DSM。

## 内存层次：从寄存器到全局内存

与线程层次对应，内存也是**分层**的：

```
thread       → 寄存器（register）：每个线程私有，最快
block        → 共享内存（shared memory）：block 内线程共享
grid         → 全局内存（global memory）：所有线程共享，最慢（但容量最大）
                 ├─ 常量内存（constant）：只读，有缓存
                 └─ 纹理/表面内存（texture/surface）：只读，有专用硬件
主机/设备之间 → 设备内存（显存）、主机内存（DRAM）
```

关键认知：

- **局部变量**默认放寄存器；放不下时溢出到"本地内存"（local memory，本质在全局内存，有 L1/L2 缓存）；
- **共享内存**是片上（on-chip）的，比全局内存快一个数量级，但容量小（现代 GPU 每 SM 几十 KB ~ 几百 KB）；
- **全局内存**访问延迟很高（数百周期），优化主要围绕它展开（第 4、10 步详述）。

## 异构编程（Host + Device）

CUDA 程序天生是**异构**的：主机（CPU）负责控制流和串行部分，设备（GPU）负责大规模数据并行部分。

```
主机线程 ──启动 kernel──▶ 设备（数千个线程并行执行）
   │                         │
   └──▶ 数据搬运（cudaMemcpy 等）
```

两个关键点：

1. **主机与设备有各自独立的内存空间**，数据要显式搬移（除非用统一内存 `__managed__`，那是另一种权衡）；
2. **kernel 内不能调用主机函数**，反之亦然；`__device__` 和 `__host__` 分别标注设备端和主机端函数（可同时标注）。

## 异步 SIMT 编程模型

Guide §5.5 专门一节讲 **Asynchronous SIMT**，这是理解"现代 CUDA"的核心：

- 传统模型：kernel 里的指令按 SIMT 方式（warp 级）执行；
- 异步模型：**异步操作**（异步拷贝、异步归约、memcpy_async 等）可以由硬件/线程块在后台推进，与当前计算**重叠执行**，不需要显式同步等待。

也就是说：**在 kernel 内部也能做异步**，不只是在主机端用 Stream 重叠。这一概念贯穿后面 CUDA Graphs、PDL、DSM 等所有现代特性。

## Compute Capability：硬件能力的标识

每个 GPU 有 `compute capability`（如 8.0 = Ampere、9.0 = Hopper、10.0 = Blackwell）：

- 它决定了支持的特性子集（如 DSM 需要 ≥9.0，Tensor Core 需要特定版本）；
- Guide 中几乎所有硬件相关描述都以它为条件；
- 编译时用 `-arch=sm_90` 指定目标，运行时可查 `cudaDeviceProp`。

> 理解"哪个特性对应哪个 capability"，是区分"变"与"不变"的钥匙。

## 小结

这一章的四个关键词：

| 概念 | 一句话 |
|---|---|
| Kernel | 设备端执行的函数，异步启动 |
| 线程层次 | grid → block →（cluster）→ thread |
| 内存层次 | 寄存器 → 共享 → 全局（+ 常量/纹理） |
| 异构模型 | 主机控制 + 设备并行，数据显式搬运 |

**编程模型是"不变"的部分**——无论 GPU 怎么迭代，这套抽象几乎没变；变的只是集群、DSM 这些在新 capability 上叠加的能力。

## 下一篇预告

第 2 步：**编程接口（Guide §6）**——kernel 是怎么被编译、加载和运行的：NVCC 编译流程、二进制/PTX 兼容性、CUDA Runtime 与驱动 API 的关系。看完你会知道 `<<<>>>` 背后到底发生了什么。
