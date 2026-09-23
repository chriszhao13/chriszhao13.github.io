---
layout: post
title: 我所理解的 CUDA · 第 1 章：Programming Model
date: 2026-08-30 10:00:00+0800
description: 《CUDA C++ Programming Guide》13.3 第 1 章 Introduction 1.2.1–1.2.3 节的中英对照翻译。
tags: cuda
featured: true
map: true
giscus_comments: true
toc:
  sidebar: left
---

> 本节对应《CUDA C++ Programming Guide》Release 13.3 **第 1 章 Programming Model，§1.2.1 Heterogeneous Systems**，采用中英对照。

> This chapter introduces the CUDA programming model at a high level and separate from any language. The terminology and concepts introduced here apply to CUDA in any supported programming language. Later chapters will illustrate these concepts in C++.

本章从较高层次介绍 CUDA 编程模型，并且不依赖于任何具体语言。本章引入的术语和概念适用于任何受支持的编程语言中的 CUDA。后续章节将用 C++ 来具体阐述这些概念。

## 1.2.1 Heterogeneous Systems（异构系统）

> The CUDA programming model assumes a heterogeneous computing system, which means a system that includes both GPUs and CPUs. The CPU and the memory directly connected to it are called the host and host memory, respectively. A GPU and the memory directly connected to it are referred to as the device and device memory, respectively. In some system-on-chip (SoC) systems, these may be part of a single package. In larger systems, there may be multiple CPUs or GPUs.

CUDA 编程模型假设系统为异构计算系统，即同时包含 GPU 和 CPU 的系统。CPU 及其直接相连的内存分别称为**主机（host）**和**主机内存（host memory）**；GPU 及其直接相连的内存分别称为**设备（device）**和**设备内存（device memory）**。在某些片上系统（SoC）中，这些组件可能集成于同一封装之内。在规模更大的系统中，则可能包含多个 CPU 或多个 GPU

> CUDA applications execute some part of their code on the GPU, but applications always start execution on the CPU. The host code, which is the code that runs on the CPU, can use CUDA APIs to copy data between the host memory and device memory, start code executing on the GPU, and wait for data copies or GPU code to complete. The CPU and GPU can both be executing code simultaneously, and best performance is usually found by maximizing utilization of both CPUs and GPUs.

CUDA 应用程序将部分代码在 GPU 上执行，但程序总是从 CPU 开始运行。主机代码（即运行在 CPU 上的代码）可通过 CUDA API 在主机内存与设备内存之间复制数据、启动 GPU 上的代码执行，并等待数据拷贝或 GPU 代码完成。CPU 与 GPU 可以同时执行各自的代码，而最佳性能通常通过最大化 CPU 与 GPU 的利用率来获得。

> The code an application executes on the GPU is referred to as device code, and a function that is invoked for execution on the GPU is, for historical reasons, called a kernel. The act of starting a kernel running is called launching the kernel. A kernel launch can be thought of as starting many threads executing the kernel code in parallel on the GPU. GPU threads operate similarly to threads on CPUs, though there are some differences important to both correctness and performance that will be covered in later sections (see Section 3.2.2.1.1).

应用程序在 GPU 上执行的代码称为**设备代码（device code）**；出于历史原因，被调用到 GPU 上执行的函数被称为**内核（kernel）**。启动一个内核运行的行为称为**启动内核（launching the kernel）**。可以把内核启动理解为：让大量线程在 GPU 上并行执行该内核代码。GPU 线程的运作方式与 CPU 线程类似，不过二者存在一些对正确性和性能都很重要的差异，这些差异将在后续章节中介绍（参见 [https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-independent-thread-scheduling](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-independent-thread-scheduling)）。(注：什么叫正确性的差异？)

---

## 1.2.2 GPU Hardware Model（GPU 硬件模型）

> Like any programming model, CUDA relies on a conceptual model of the underlying hardware. For the purposes of CUDA programming, the GPU can be considered to be a collection of Streaming Multiprocessors (SMs) which are organized into groups called Graphics Processing Clusters (GPCs). Each SM contains a local register file, a unified data cache, and a number of functional units that perform computations. The unified data cache provides the physical resources for shared memory and L1 cache. The allocation of the unified data cache to L1 and shared memory can be configured at runtime. The sizes of different types of memory and the number of functional units within an SM can vary across GPU architectures.

与任何编程模型一样，CUDA 也建立在底层硬件的一个概念模型之上（CUDA 的抽象是对硬件的一种简化视角，方便程序员不关心具体微架构细节也能编程）。就 CUDA 编程而言，可以把 GPU 看作一组**流式多处理器（Streaming Multiprocessors，简称 SM）**的集合，这些 SM 被组织成称为**图形处理簇（Graphics Processing Clusters，简称 GPC）**的分组。每个 SM 包含一个本地寄存器文件、一个统一数据缓存，以及若干执行计算的功能单元。统一数据缓存为共享内存和 L1 缓存提供物理资源。**统一数据缓存在 L1 与共享内存之间的分配比例可以在运行时配置**。不同类型内存的容量以及 SM 内功能单元的数量会因 GPU 架构而异。

> Note: The actual hardware layout of a GPU or the way it physically carries out the execution of the programming model may vary. These differences do not affect correctness of software written using the CUDA programming model.

**注**：GPU 的实际硬件布局，或它在物理层面执行该编程模型的具体方式，可能会有所不同。但这些差异不影响使用 CUDA 编程模型编写的软件的正确性。

{% include figure.liquid loading="eager" path="assets/img/cuda101/image02.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
    图 2：GPU 包含许多流式多处理器（SM），每个 SM 内部又包含大量功能单元。图形处理簇（GPC）是 SM 的集合。一个 GPU 由一组连接到 GPU 内存的 GPC 组成。CPU 通常具有多个核心，以及一个连接到系统内存的内存控制器。CPU 与 GPU 之间通过诸如 PCIe 或 NVLink 之类的互连进行连接。
</div>
---

### 1.2.2.1. Thread Blocks and Grids

> When an application launches a kernel, it does so with many threads, often millions of threads. These threads are organized into blocks. A block of threads is referred to, perhaps unsurprisingly, as a thread block. Thread blocks are organized into a grid. All the thread blocks in a grid have the same size and dimensions. Figure 3 shows an illustration of a grid of thread blocks.

当应用程序启动一个内核时，会同时启动大量线程，通常多达数百万个。这些线程被组织成**块（block）**。一个线程块被称为**线程块（thread block）**—— 这个名字或许并不令人意外。线程块进一步被组织成**网格（grid）**。一个网格中的所有线程块具有相同的大小和维度。图 3 展示了一个由线程块组成的网格示意图。

{% include figure.liquid loading="eager" path="assets/img/cuda101/image03.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
    图 3：线程块网格。每个箭头代表一个线程（箭头数量并不代表实际线程数量）。
</div>

> Thread blocks and grids may be 1, 2, or 3 dimensional. These dimensions can simplify mapping of individual threads to units of work or data items.

线程块和网格可以是一维、二维或三维的。这些维度可以简化将单个线程映射到工作单元或数据项的过程。

> When a kernel is launched, it is launched using a specific execution configuration which specifies the grid and thread block dimensions. The execution configuration may also include optional parameters such as cluster size, stream, and SM configuration settings, which will be introduced in later sections.

启动内核时，需要使用一个特定的**执行配置（execution configuration）**，其中指定了网格和线程块的维度。执行配置还可以包含一些可选参数，例如**簇大小（cluster size）**、**流（stream）**以及 **SM 配置设置**等，这些内容将在后续章节中介绍。

注：**"cluster size"** → "簇大小"，指 thread block cluster（线程块簇），这是 Hopper 架构引入的新特性，允许多个 block 组成一个 cluster 协同执行，共享分布式共享内存。属于较新的进阶内容。

> Using built-in variables, each thread executing the kernel can determine its location within its containing block and the location of its block within the containing grid. A thread can also use these built-in variables to determine the dimensions of the thread blocks and the grid on which the kernel was launched. This gives each thread a unique identity among all the threads running the kernel. This identity is frequently used to determine what data or operations a thread is responsible for.

通过使用内置变量，执行内核的每个线程都可以确定自己在所属线程块中的位置，以及其所属线程块在所属网格中的位置。线程还可以利用这些内置变量确定内核启动时所使用的线程块和网格的维度。这使得每个线程在所有运行该内核的线程中都拥有一个唯一标识。该标识通常用于确定一个线程负责处理哪些数据或执行哪些操作。

> All threads of a thread block are executed in a single SM. This allows threads within a thread block to communicate and synchronize with each other efficiently. Threads within a thread block all have access to the on-chip shared memory, which can be used for exchanging information between threads of a thread block.

一个线程块中的所有线程都在**同一个 SM 上执行**。这使得线程块内的线程能够高效地相互通信和同步。线程块内的所有线程都可以访问**片上共享内存（on-chip shared memory）**，该内存可用于线程块内线程之间交换信息。

> A grid may consist of millions of thread blocks, while the GPU executing the grid may have only tens or hundreds of SMs. All threads of a thread block are executed by a single SM and, in most cases [1], run to completion on that SM. There is no guarantee of scheduling between thread blocks, so a thread block cannot rely on results from other thread blocks, as they may not be able to be scheduled until that thread block has completed. Figure 4 shows an example of how thread blocks from a grid are assigned to an SM.

一个网格可能包含数百万个线程块，而执行该网格的 GPU 可能只有几十个或上百个 SM。一个线程块的所有线程都由同一个 SM 执行，并且在大多数情况下 [1] 会在该 SM 上运行至结束。线程块之间的调度没有任何保证，因此一个线程块不能依赖其他线程块的结果 —— 因为那些线程块可能要等到当前线程块完成后才能被调度。图 4 展示了网格中的线程块如何被分配到 SM 上的一个示例。

[1] In certain situations when using features such as [https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism) , a thread block may be suspended to memory. This means the state of the SM is stored to a system-managed area of GPU memory and the SM is freed to execute other thread blocks. This is similar to context swapping on CPUs. This is not common.

[1] 在某些情况下，当使用诸如**动态并行（dynamic parallelism）**（[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism)）等特性时，一个线程块可能会被**挂起到内存（suspended to memory）**。这意味着该 SM 的状态被保存到 GPU 内存中由系统管理的区域，同时该 SM 被释放出来以执行其他线程块。这类似于 CPU 上的**上下文切换（context swapping）**。这种情况并不常见。

{% include figure.liquid loading="eager" path="assets/img/cuda101/image04.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
    图 4：每个 SM 有一个或多个活跃的线程块。在本示例中，每个 SM 同时调度了三个线程块。网格中的线程块以何种顺序分配到各个 SM，没有任何保证。
</div>

> The CUDA programming model enables arbitrarily large grids to run on GPUs of any size, whether it has only one SM or thousands of SMs. To achieve this, the CUDA programming model, with some exceptions, requires that there be no data dependencies between threads in different thread blocks. That is, a thread should not depend on results from or synchronize with a thread in a different thread block of the same grid. All the threads within a thread block run on the same SM at the same time. Different thread blocks within the grid are scheduled among the available SMs and may be executed in any order. In short, the CUDA programming model requires that it be possible to execute thread blocks in any order, in parallel or in series.

CUDA 编程模型使得任意大小的网格都能在任何规模的 GPU 上运行 —— 无论该 GPU 只有一个 SM，还是拥有数千个 SM。为了实现这一点，CUDA 编程模型（除少数例外情况外）要求**不同线程块中的线程之间不存在数据依赖**。也就是说，一个线程不应依赖同一网格中另一个线程块内线程的结果，也不应与之同步。一个线程块内的所有线程同时在同一个 SM 上运行。网格中的不同线程块被调度到可用的 SM 上，并且可以以任意顺序执行。简而言之，CUDA 编程模型要求：线程块必须能够以任意顺序执行，无论是并行执行还是串行执行。

### 1.2.2.1.1. Thread Block Clusters

> In addition to thread blocks, GPUs with compute capability 9.0 and higher have an optional level of grouping called clusters. Clusters are a group of thread blocks which, like thread blocks and grids, can be laid out in 1, 2, or 3 dimensions. [https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#figure-thread-block-clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#figure-thread-block-clusters) illustrates a grid of thread blocks that is also organized into clusters. Specifying clusters does not change the grid dimensions or the indices of a thread block within a grid.

除线程块之外，计算能力（compute capability）为 9.0 及更高版本的 GPU 还提供了一个可选的分组层级，称为**簇（cluster）**。簇是一组线程块的集合，与线程块和网格一样，也可以按一维、二维或三维布局。[https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#figure-thread-block-clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#figure-thread-block-clusters) 展示了一个同时被组织为簇的线程块网格示意图。指定簇并不会改变网格的维度，也不会改变线程块在网格中的索引。

{% include figure.liquid loading="eager" path="assets/img/cuda101/image05.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
    图 5：当指定了簇时，线程块在网格中仍处于原来的位置，但同时也拥有其在所属簇中的位置。
</div>

> Specifying clusters groups adjacent thread blocks into clusters and provides some additional opportunities for synchronization and communication at the cluster level. Specifically, all thread blocks in a cluster are executed in a single GPC. Figure 6 shows how thread blocks are scheduled to SMs in a GPC when clusters are specified. Because the thread blocks are scheduled simultaneously and within a single GPC, threads in different blocks but within the same cluster can communicate and synchronize with each other using software interfaces provided by Cooperative Groups. Threads in clusters can access the shared memory of all blocks in the cluster, which is referred to as distributed shared memory.The maximum size of a cluster is hardware dependent and varies between devices.

指定簇会将相邻的线程块分组到各个簇中，并在簇级别提供一些额外的同步与通信机会。具体而言，一个簇中的所有线程块都在**同一个 GPC 内执行**。[https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-scheduling-with-clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-scheduling-with-clusters) 展示了指定簇时，线程块如何被调度到一个 GPC 内的各个 SM 上。由于这些线程块是同时被调度的，且位于同一个 GPC 内，因此**不同 block 但属于同一簇的线程**可以通过 [https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-cooperative-groups](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-cooperative-groups) 提供的软件接口相互通信和同步。簇中的线程可以访问该簇内所有 block 的共享内存，这被称为**分布式共享内存（distributed shared memory）**（[https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-distributed-shared-memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-distributed-shared-memory)）。簇的最大大小取决于硬件，因设备而异。

> Figure 6 illustrates the how thread blocks within a cluster are scheduled simultaneously on SMs within a GPC. Thread blocks within a cluster are always adjacent to each other within the grid.

图 6 展示了一个簇内的线程块如何同时被调度到一个 GPC 内的各个 SM 上。一个簇内的线程块在网格中总是彼此相邻的。

{% include figure.liquid loading="eager" path="assets/img/cuda101/image06.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
    图 6：当指定簇时，一个簇内的线程块被同时调度到一个 GPC 内的各个 SM 上。一个簇内的线程块在网格中总是彼此相邻的。
</div>

### 1.2.2.2. Warps and SIMT

> Within a thread block, threads are organized into groups of 32 threads called warps. A warp executes the kernel code in a Single-Instruction Multiple-Threads (SIMT) paradigm. In SIMT, all threads in the warp are executing the same kernel code, but each thread may follow different branches through the code. That is, though all threads of the program execute the same code, threads do not need to follow the same execution path.

在线程块内部，线程被组织成由 32 个线程组成的组，称为**线程束（warp）**。一个 warp 以**单指令多线程（Single-Instruction Multiple-Threads，简称 SIMT）**范式执行内核代码。在 SIMT 中，warp 内的所有线程执行同一份内核代码，但每个线程可能在代码中走不同的分支。也就是说，尽管程序的所有线程执行同一份代码，但线程不必遵循相同的执行路径。

> When threads are executed by a warp, they are assigned a warp lane. Warp lanes are numbered 0 to 31 and threads from a thread block are assigned to warps in a predictable fashion detailed in Hardware Multithreading.

当线程由一个 warp 执行时，它们会被分配一个 **warp lane（warp 通道）**。Warp lane 的编号为 0 到 31，线程块中的线程以一种可预测的方式被分配到各个 warp 中，具体方式详见 "硬件多线程（Hardware Multithreading）" 章节。

> All threads in the warp execute the same instruction simultaneously. If some threads within a warp follow a control flow branch in execution while others do not, the threads which do not follow the branch will be masked off while the threads which follow the branch are executed. For example, if a conditional is only true for half the threads in a warp, the other half of the warp would be masked off while the active threads execute those instructions. This situation is illustrated in Figure 7. When different threads in a warp follow different code paths, this is sometimes called warp divergence. It follows that utilization of the GPU is maximized when threads within a warp follow the same control flow path.

Warp 中的所有线程同时执行同一条指令。如果 warp 内的某些线程在执行中走了某个控制流分支，而其他线程没有，那么在执行该分支的指令时，没有走该分支的线程会被**屏蔽（masked off）**。例如，如果一个条件只对 warp 中一半的线程为真，那么在活跃线程执行这些指令时，warp 的另一半会被屏蔽。这种情况如 图 7 所示。当 warp 内的不同线程走不同的代码路径时，这种情况有时被称为 **warp 发散（warp divergence）**。由此可知，当 warp 内的线程走相同的控制流路径时，GPU 的利用率最高。

{% include figure.liquid loading="eager" path="assets/img/cuda101/image07.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
    图 7：在本示例中，只有线程索引为偶数的线程执行 if 语句的主体，在执行该主体时，其余线程被屏蔽。
</div>

> In the SIMT model, all threads in a warp progress through the kernel in lock step. Hardware execution may differ. See the sections on Independent Thread Execution for more information on where this distinction is important. Exploiting knowledge of how warp execution is actually mapped to real hardware is discouraged. The CUDA programming model and SIMT say that all threads in a warp progress through the code together. Hardware may optimize masked lanes in ways that are transparent to the program so long as the programming model is followed. If the program violates this model, this can result in undefined behavior that can be different in different GPU hardware.

在 SIMT 模型中，warp 内的所有线程以**锁步（lock step）**方式推进内核的执行。硬件的实际执行方式可能与此不同。关于这一区别在哪些情况下很重要，更多信息请参见 [https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-independent-thread-scheduling](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-independent-thread-scheduling) 章节。**不建议**利用 "warp 执行实际上如何映射到真实硬件" 的知识来编程。CUDA 编程模型和 SIMT 规定：warp 内的所有线程共同推进代码执行。只要遵循编程模型，硬件可能以对程序透明的方式优化被屏蔽的 lane。如果程序违反了这一模型，可能导致**未定义行为（undefined behavior）**，且在不同的 GPU 硬件上表现可能不同。

> While it is not necessary to consider warps when writing CUDA code, understanding the warp execution model is helpful in understanding concepts such as global memory coalescing and shared memory bank access patterns. Some advanced programming techniques use specialization of warps within a thread block to limit thread divergence and maximize utilization. This and other optimizations make use of the knowledge that threads are grouped into warps when executing.

虽然编写 CUDA 代码时**不必**考虑 warp，但理解 warp 执行模型有助于理解诸如 [https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-coalesced-global-memory-access](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-coalesced-global-memory-access) （合并的全局内存访问）和 [https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-shared-memory-access-patterns](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-shared-memory-access-patterns) （共享内存访问模式）等概念。一些高级编程技术会在线程块内对 warp 进行**专门化（specialization）**，以限制线程发散并最大化利用率。这类优化以及其他优化都利用了 "线程在执行时被分组为 warp" 这一知识。

> One implication of warp execution is that thread blocks are best specified to have a total number of threads which is a multiple of 32. It is legal to use any number of threads, but when the total is not a multiple of 32, the last warp of the thread block will have some lanes that are unused throughout execution. This will likely lead to suboptimal functional units utilization and memory access for that warp.

Warp 执行的一个推论是：线程块的总线程数**最好指定为 32 的倍数**。使用任意数量的线程都是合法的，但当总数不是 32 的倍数时，线程块的最后一个 warp 会有一些 lane 在整个执行过程中未被使用。这可能导致该 warp 的功能单元利用率和内存访问效率不佳。

> SIMT is often compared to Single Instruction Multiple Data (SIMD) parallelism, but there are some important differences. In SIMD, execution follows a single control flow path, while in SIMT, each thread is allowed to follow its own control flow path. Because of this, SIMT does not have a fixed data-width like SIMD. A more detailed discussion of SIMT can be found in SIMT Execution Model.

SIMT 常被拿来与**单指令多数据（SIMD）**并行相比较，但二者存在一些重要区别。在 SIMD 中，执行遵循**单一控制流路径**；而在 SIMT 中，每个线程都被允许走**自己的控制流路径**。正因如此，SIMT 不像 SIMD 那样具有固定的数据宽度。关于 SIMT 更详细的讨论可参见 [https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-simt-architecture](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-simt-architecture) 。

### 1.2.2.3. Tile Programming in CUDA
