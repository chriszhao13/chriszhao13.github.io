---
layout: post
title: 我所理解的 CUDA · 开篇
date: 2026-08-11 10:00:00+0800
description: 一个以 CUDA C++ Programming Guide Release 13.3 为纲，从基本原理到最新特性逐步深入的 CUDA 学习系列。
tags: cuda
featured: false
---

> 本系列以 NVIDIA 官方 **[《CUDA C++ Programming Guide》Release 13.3](https://docs.nvidia.com/cuda/cuda-programming-guide/contents.html)**（2026 年 5 月发布）为唯一权威依据，用中文逐步解读：先从基本原理建立模型，再走到最新特性，配合可执行的代码示例加深理解。

## 为什么以 Programming Guide 为纲

GPU 与 CUDA 的资料浩如烟海，博客、教程、白皮书各有侧重，但**官方 Programming Guide 是唯一一份从编程模型一路讲到硬件实现、且全链路自洽**的文档。作为 GPU 加速计算领域的研究人员，它迟早是要完整走一遍的。这个系列便为此而写——以写促学，逼着自己把每一章真正读懂、想透，而不是浅尝辄止。

选择它，还因为它有几点不可替代：

- **概念稳定**——它讲的是基本概念（线程层次、内存层次、SIMT），这些几乎不随架构迭代而变化；
- **特性最新**——它同步跟踪最新特性（13.3 已覆盖 Thread Block Clusters、分布式共享内存、CUDA Graphs 条件节点、设备端图启动等）；
- **可作坐标**——它是理解其他一切资料（Best Practices、架构白皮书、技术博客）的坐标系。

## 系列规划

| 篇章 | 主题 |
|---|---|
| 开篇 | 为什么跟着 Programming Guide 学 CUDA（本篇） |
| 第 1 章 | Chapter 1. Introduction to CUDA |

---

*这是用来给我自己学习的记录，如果理解有误欢迎讨论。*
