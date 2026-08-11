---
layout: post
title: 我所理解的 CUDA · 第 1 章：Introduction（1.1.1 The Graphics Processing Unit）
date: 2026-08-25 10:00:00+0800
description: 《CUDA C++ Programming Guide》13.3 第 1 章 Introduction 1.1.1 节的中文翻译。
tags: cuda
featured: false
---

> 本节对应《CUDA C++ Programming Guide》Release 13.3 **第 1 章 Introduction，§1.1.1 The Graphics Processing Unit** 的中文翻译。

## 1.1.1 图形处理单元（The Graphics Processing Unit）

图形处理单元（Graphics Processing Unit，GPU）诞生之初是为 3D 图形而生的专用处理器，最初以固定功能（fixed-function）硬件的形式，加速实时 3D 渲染中的并行操作。随着一代又一代的演进，GPU 变得越来越可编程。到 2003 年，图形管线中的某些阶段已经可以完全可编程，能够针对 3D 场景或图像中的每一个分量并行执行自定义代码。

2006 年，NVIDIA 推出了计算统一设备架构（Compute Unified Device Architecture，CUDA），使任何计算负载都能独立于图形 API，利用 GPU 的吞吐能力（throughput capability）。

自此以后，CUDA 与 GPU 计算被用于加速几乎所有类型的计算负载：从流体动力学、能量输运等科学模拟，到数据库、数据分析等商业应用。此外，GPU 的能力与可编程性，已成为从图像分类到生成式人工智能（如扩散模型、大语言模型）等新算法与新技术进步的基础（foundational）。
