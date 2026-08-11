---
layout: post
title: 我所理解的 CUDA · 第 1 章：Introduction（1.1.1–1.1.3）
date: 2026-08-11 10:00:00+0800
description: 《CUDA C++ Programming Guide》13.3 第 1 章 Introduction 1.1.1–1.1.3 节的中英对照翻译。
tags: cuda
featured: true
map: true
disqus_comments: true
toc:
  sidebar: left
---

> 本节对应《CUDA C++ Programming Guide》Release 13.3 **第 1 章 Introduction，§1.1.1 The Graphics Processing Unit**，采用中英对照。

## 1.1.1 The Graphics Processing Unit（图形处理单元）

> Born as a special-purpose processor for 3D graphics, the Graphics Processing Unit (GPU) started out as fixed-function hardware to accelerate parallel operations in real-time 3D rendering. Over successive generations, GPUs became more programmable. By 2003, some stages of the graphics pipeline became fully programmable, running custom code in parallel for each component of a 3D scene or an image.

图形处理单元（Graphics Processing Unit，GPU）诞生之初是为 3D 图形而生的专用处理器，最初以固定功能（fixed-function）硬件的形式，加速实时 3D 渲染中的并行操作。随着一代又一代的演进，GPU 变得越来越可编程。到 2003 年，图形管线中的某些阶段已经可以完全可编程，能够针对 3D 场景或图像中的每一个分量并行执行自定义代码。

> In 2006, NVIDIA introduced the Compute Unified Device Architecture (CUDA) to enable any computational workload to use the throughput capability of GPUs independent of graphics APIs.

2006 年，NVIDIA 推出了计算统一设备架构（Compute Unified Device Architecture，CUDA），使任何计算负载都能独立于图形 API，利用 GPU 的吞吐能力（throughput capability）。

> Since then, CUDA and GPU computing have been used to accelerate computational workloads of nearly every type, from scientific simulations such as fluid dynamics or energy transport to business applications like databases and analytics. Moreover, the capability and programmability of GPUs has been foundational to the advancement of new algorithms and technologies ranging from image classification to generative artificial intelligence such as diffusion or large language models.

自此以后，CUDA 与 GPU 计算被用于加速几乎所有类型的计算负载：从流体动力学、能量输运等科学模拟，到数据库、数据分析等商业应用。此外，GPU 的能力与可编程性，已成为从图像分类到生成式人工智能（如扩散模型、大语言模型）等新算法与新技术进步的基础（foundational）。

---

> 本节对应《CUDA C++ Programming Guide》Release 13.3 **第 1 章 Introduction，§1.1.2 The Benefits of Using GPUs**，采用中英对照。

## 1.1.2 The Benefits of Using GPUs（使用 GPU 的收益）

> A GPU provides much higher instruction throughput and memory bandwidth than a CPU within a similar price and power envelope. Many applications leverage these capabilities to run significantly faster on the GPU than on the CPU (see [GPU Applications](https://www.nvidia.com/en-us/accelerated-applications/)). Other computing devices, like FPGAs, are also very energy efficient, but offer much less programming flexibility than GPUs.

GPU 在相近的价格与功耗范围（price and power envelope）内，能提供比 CPU 高得多的指令吞吐量（instruction throughput）与内存带宽（memory bandwidth）。许多应用利用这些能力，在 GPU 上比在 CPU 上运行得明显更快（参见 [GPU Applications](https://www.nvidia.com/en-us/accelerated-applications/)）。其他计算设备，如 FPGA，也非常节能，但相比 GPU，其编程灵活性（programming flexibility）要低得多。

> GPUs and CPUs are designed with different goals in mind. While a CPU is designed to excel at executing a serial sequence of operations (called a thread) as fast as possible and can execute a few tens of these threads in parallel, a GPU is designed to excel at executing thousands of threads in parallel, trading off lower single-thread performance to achieve much greater total throughput.

GPU 与 CPU 的设计目标各不相同。CPU 的设计目标是尽可能快地执行串行操作序列（称为线程 thread），并能并行执行数十个这样的线程；而 GPU 的设计目标则是并行执行数千个线程，以较低的单线程性能（single-thread performance）为代价，换取大得多的总吞吐量（total throughput）。

> GPUs are specialized for highly parallel computations and devote more transistors to data processing units, while CPUs dedicate more transistors to data caching and flow control. [Figure 1](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html#from-graphics-processing-to-general-purpose-parallel-computing-gpu-devotes-more-transistors-to-data-processing) shows an example distribution of chip resources for a CPU versus a GPU.

GPU 专精于高度并行的计算，将更多晶体管（transistors）投入到数据处理单元；而 CPU 则将更多晶体管用于数据缓存（data caching）与流控制（flow control）。[图 1](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html#from-graphics-processing-to-general-purpose-parallel-computing-gpu-devotes-more-transistors-to-data-processing) 展示了 CPU 与 GPU 芯片资源分布的一个示例。

{% include figure.liquid loading="eager" path="assets/cuda101/image01.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
    图 1：GPU 将更多晶体管投入到数据处理
</div>

---

> 本节对应《CUDA C++ Programming Guide》Release 13.3 **第 1 章 Introduction，§1.1.3 Getting Started Quickly**，采用中英对照。

## 1.1.3 Getting Started Quickly（快速上手）

> There are many ways to leverage the compute power provided by GPUs. This guide covers programming for the CUDA GPU platform in high-level languages such as C++. However, there are many ways to utilize GPUs in applications that do not require directly writing GPU code.

利用 GPU 提供的计算能力有多种方式。本指南介绍的是用 C++ 等高级语言为 CUDA GPU 平台编程的方法。不过，在不直接编写 GPU 代码的情况下，也有很多方式可以在应用中使用 GPU。

> An ever-growing collection of algorithms and routines from a variety of domains is available through specialized libraries. When a library has already been implemented—especially those provided by NVIDIA—using it is often more productive and performant than reimplementing algorithms from scratch. Libraries like cuBLAS, cuFFT, cuDNN, and CUTLASS are just a few examples of libraries that help developers avoid reimplementing well-established algorithms. These libraries have the added benefit of being optimized for each GPU architecture, providing an ideal mix of productivity, performance, and portability.

来自众多领域、且不断增长的算法与例程（routines）（CZ: 这个是个固定术语）集合，可通过专用库（specialized libraries）获取。当某个库已经被实现出来时——尤其是 NVIDIA 提供的库——使用它往往比从头重新实现算法更高效、性能更好。cuBLAS、cuFFT、cuDNN 与 CUTLASS 等库，只是帮助开发者避免重复实现成熟算法的少数几个例子。这些库还有一项额外的好处：它们针对每一种 GPU 架构都做了优化，从而在生产效率（productivity）、性能（performance）与可移植性（portability）之间提供了理想的组合。

> There are also frameworks, particularly those used for artificial intelligence, that provide GPU-accelerated building blocks. Many of these frameworks achieve their acceleration by leveraging the GPU-accelerated libraries mentioned above.

还有一些框架，尤其是用于人工智能的框架，提供 GPU 加速的构建模块（building blocks）。其中许多框架正是通过利用上述 GPU 加速库来实现其加速效果的。

> Additionally, domain-specific languages (DSLs) such as NVIDIA’s Warp or OpenAI’s Triton compile to run directly on the CUDA platform. This provides an even higher-level method of programming GPUs than the high-level languages covered in this guide.

此外，诸如 NVIDIA 的 Warp、OpenAI 的 Triton 等领域特定语言（domain-specific languages，DSLs），会被编译为直接在 CUDA 平台上运行。与指南所涵盖的高级语言相比，这提供了一种更高级的 GPU 编程方式。

> The [NVIDIA Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub) contains resources, examples, and tutorials to teach GPU and CUDA computing.

[NVIDIA Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub) 收录了用于教授 GPU 与 CUDA 计算的资源、示例和教程。
