---
layout: post
title: 我所理解的 CUDA · 开篇：跟着 Programming Guide 学 CUDA
date: 2026-08-11 10:00:00+0800
description: 一个以 CUDA C++ Programming Guide Release 13.3 为纲，从基本原理到最新特性逐步深入的 CUDA 学习系列。
tags: cuda
featured: false
---

> 本系列以 NVIDIA 官方 **《CUDA C++ Programming Guide》Release 13.3**（2026 年 5 月发布）为唯一权威依据，用中文逐步解读：先从基本原理建立模型，再走到最新特性，配合代码示例加深理解。

## 为什么以 Programming Guide 为纲

GPU 与 CUDA 的资料浩如烟海，但**官方 Programming Guide 是唯一一份"从模型到实现"全链路一致**的文档：

- 它讲的是**稳定的抽象模型**（线程层次、内存层次、SIMT），这些几乎不随架构变化；
- 它同步跟踪**最新特性**（13.3 已覆盖 Thread Block Clusters、分布式共享内存、CUDA Graphs 条件节点、设备端图启动等）；
- 它是理解其他一切资料（Best Practices、架构白皮书、博客）的**坐标系**。

## 系列规划

| 篇章 | 主题 |
|---|---|
| 开篇 | 为什么跟着 Programming Guide 学 CUDA（本篇） |
| 第 1 步 | 编程模型：Kernel、线程层次、内存层次 |
| 第 2 步 | 异构编程与异步 SIMT 模型 |
| 第 3 步 | 编译流程：NVCC、离线/JIT、二进制与 PTX 兼容性 |
| 第 4 步 | 设备内存与 L2 访问管理 |
| 第 5 步 | 共享内存与分布式共享内存（DSM） |
| 第 6 步 | 异步并发执行：Stream、事件、PDL |
| 第 7 步 | CUDA Graphs：图结构、Stream Capture、设备端图启动 |
| 第 8 步 | 多设备、统一虚拟地址与进程间通信 |
| 第 9 步 | 硬件实现：SIMT 架构与硬件多线程 |
| 第 10 步 | 性能指南：占用率、访存与指令吞吐 |
| 第 11 步 | 语言扩展（上）：执行空间、变量空间、内置变量 |
| 第 12 步 | 语言扩展（下）：原子操作与 Warp 级原语 |

## 下一篇预告

第 1 步：**编程模型**——从最简单的一个 Kernel 出发，逐步拆解线程层次（grid → block → warp → thread）、内存层次（寄存器 → 共享 → 全局）以及 Thread Block Clusters 这一较新的抽象。这是整个 Guide 的地基。

---

*这是用来给我自己学习的记录，如果理解有误欢迎讨论。*
