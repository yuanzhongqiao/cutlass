<div class="Box-sc-g0xbh4-0 bJMeLZ js-snippet-clipboard-copy-unpositioned" data-hpc="true"><article class="markdown-body entry-content container-lg" itemprop="text"><p dir="auto"><a target="_blank" rel="noopener noreferrer" href="/NVIDIA/cutlass/blob/main/media/images/gemm-hierarchy-with-epilogue-no-labels.png"><img src="/NVIDIA/cutlass/raw/main/media/images/gemm-hierarchy-with-epilogue-no-labels.png" alt="丙氨酸转氨酶" title="完整的 CUDA GEMM 分解" style="max-width: 100%;"></a></p>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">弯刀3.5</font></font></h1><a id="user-content-cutlass-35" class="anchor" aria-label="永久链接：CUTLASS 3.5" href="#cutlass-35"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">弯刀 3.5 - 2024 年 4 月</font></font></em></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 是 CUDA C++ 模板抽象的集合，用于在 CUDA 内的所有级别和规模上实现高性能矩阵-矩阵乘法 (GEMM) 和相关计算。它包含类似于用于实现 cuBLAS 和 cuDNN 的分层分解和数据移动策略。 CUTLASS 将这些“移动部件”分解为由 C++ 模板类抽象的可重用、模块化软件组件。概念并行化层次结构不同级别的原语可以通过自定义平铺大小、数据类型和其他算法策略进行专门化和调整。由此产生的灵活性简化了它们在自定义内核和应用程序中作为构建块的使用。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">为了支持各种应用，CUTLASS 为混合精度计算提供了广泛的支持，为半精度浮点 (FP16)、BFloat16 (BF16)、Tensor Float 32 (TF32)、单精度浮点 (FP32)、
</font></font><a href="/NVIDIA/cutlass/blob/main/examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">通过张量核心指令进行 FP32 仿真</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">、双精度浮点 (FP64) 类型、整数数据类型（4b 和 8b）以及二进制数据类型 (1b)。 CUTLASS 演示了针对</font><font style="vertical-align: inherit;">由 NVIDIA Volta、Turing、Ampere 和 Hopper 架构实现的可编程、高吞吐量</font></font><em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Tensor Core 的</font></font></em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">曲速同步矩阵乘法运算。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">请参阅</font></font><a href="/NVIDIA/cutlass/blob/main/media/docs/quickstart.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">快速入门指南</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">以快速开始。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">请参阅</font></font><a href="/NVIDIA/cutlass/blob/main/media/docs/functionality.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">功能列表</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">，了解执行模型层次结构的每个级别支持的操作列表。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 3.0 引入了一个新的核心库 CuTe，用于描述和操作线程和数据的张量。 CuTe 是 C++ CUDA 模板抽象的集合，用于定义和操作线程和数据的分层多维布局。 CuTe 提供了紧凑地封装数据的类型、形状、内存空间和布局的</font></font><code>Layout</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">对象</font></font><code>Tensor</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">，同时为用户执行复杂的索引。这让程序员可以专注于算法的逻辑描述，而 CuTe 则为他们进行机械记账。借助这些工具，我们可以快速设计、实现和修改所有密集线性代数运算。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CuTe 的核心抽象是分层多维布局，可以用数据数组组成来表示张量。布局的表示足够强大，足以表示我们实现高效密集线性代数所需的几乎所有内容。布局还可以通过功能组合来组合和操作，在此基础上我们构建了大量常见操作，例如平铺和分区。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 3.0 及更高版本在其模板的整个 GEMM 层次结构中采用 CuTe。这极大地简化了设计并提高了代码的可组合性和可读性。更多特定于 CuTe 的文档可以在其</font></font><a href="/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">专用文档目录</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">中找到</font><font style="vertical-align: inherit;">。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">除了 GEMM 之外，CUTLASS 通过隐式 GEMM 算法实现高性能卷积。隐式 GEMM 是将卷积运算表述为 GEMM，从而利用 CUTLASS 的模块化 GEMM 管道。这使得 CUTLASS 能够通过重用高度优化的 GEMM 组件来构建卷积。</font></font></p>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 3.5 的新增功能</font></font></h1><a id="user-content-whats-new-in-cutlass-35" class="anchor" aria-label="永久链接：CUTLASS 3.5 的新增功能" href="#whats-new-in-cutlass-35"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 3.5 是 CUTLASS 的更新，添加了：</font></font></p>
<ul dir="auto">
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">通过 WGMMA + </font></font><a href="/NVIDIA/cutlass/blob/main/include/cute/atom/copy_traits_sm90_im2col.hpp"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">TMA im2col</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">针对 Hopper SM90A 的隐式 GEMM 卷积。
</font></font><ul dir="auto">
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">使用 CuTe 在 CUTLASS 3.x 中进行本机实现，镜像与</font></font><a href="/NVIDIA/cutlass/blob/main/media/docs/gemm_api_3x.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">GEMM 相同的设计层次结构</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。</font></font></li>
<li><font style="vertical-align: inherit;"></font><a href="/NVIDIA/cutlass/blob/main/include/cutlass/conv/convnd_problem_shape.hpp"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">以与等级无关的方式</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">支持 1D、2D 和 3D 卷积</font><font style="vertical-align: inherit;">。</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">支持</font></font><a href="/NVIDIA/cutlass/blob/main/test/unit/conv/device_3x/fprop/sm90_conv3d_fprop_implicit_gemm_s8_s8_s32_tensorop_s32.cu"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Fprop</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">、</font></font><a href="/NVIDIA/cutlass/blob/main/test/unit/conv/device_3x/dgrad/sm90_conv2d_dgrad_implicit_gemm_f16_f16_f32_tensorop_f16.cu"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Dgrad</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">和</font></font><a href="/NVIDIA/cutlass/blob/main/test/unit/conv/device_3x/wgrad/sm90_conv1d_wgrad_implicit_gemm_f16_f16_f32_tensorop_f16.cu"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Wgrad</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">算法。</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/python/cutlass_library/conv3x_emitter.py"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 分析器支持</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">通过 3.x API 实现的 2D 和 3D 卷积。</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">注意：这是测试版。 CUTLASS 的进一步更新将包括重大性能改进、功能启用以及在 3.7 版本发布之前可能对 API 进行的重大更改。欢迎您对设计提出反馈！</font></font></li>
</ul>
</li>
<li><font style="vertical-align: inherit;"></font><a href="/NVIDIA/cutlass/blob/main/examples/58_ada_fp8_gemm/ada_fp8_gemm.cu"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">通过 2.x API</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">支持Ada (SM89) FP8 张量核心</font><font style="vertical-align: inherit;">。需要 CUDA 12.4 或更高版本。</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/examples/59_ampere_gather_scatter_gemm/README.md"><font style="vertical-align: inherit;"></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CuTe 和 CUTLASS 3.x 中的
</font><a href="/NVIDIA/cutlass/blob/main/examples/59_ampere_gather_scatter_gemm/README.md"><font style="vertical-align: inherit;">安培聚集/散射卷积示例。</font></a></font><ul dir="auto">
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">展示如何使用 CUTLASS 3.x 和 CuTe 编写和优化自定义内核，以及将卷积实现为 GETT 专业化的一般策略。</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">实施粗粒度稀疏聚集/分散内核，在安培级张量核心上实现峰值性能。</font></font></li>
</ul>
</li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 2.x 中添加了 32x 和 16x 切片尺寸，以提高窄高和宽短矩阵的性能。</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">更新了</font><a href="/NVIDIA/cutlass/blob/main/media/docs/cute/0t_mma_atom.md"><font style="vertical-align: inherit;">MMA 原子</font></a></font><a href="/NVIDIA/cutlass/blob/main/media/docs/cute/03_tensor.md"><code>cute::Tensor&lt;&gt;</code></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">的 CuTe 文档，</font><font style="vertical-align: inherit;">以及经过彻底修改的</font><a href="/NVIDIA/cutlass/blob/main/examples/cute/tutorial"><font style="vertical-align: inherit;">CuTe GEMM 教程系列</font></a><font style="vertical-align: inherit;">。</font></font><a href="/NVIDIA/cutlass/blob/main/media/docs/cute/0t_mma_atom.md"><font style="vertical-align: inherit;"></font></a><font style="vertical-align: inherit;"></font><a href="/NVIDIA/cutlass/blob/main/examples/cute/tutorial"><font style="vertical-align: inherit;"></font></a><font style="vertical-align: inherit;"></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CuTe 的扩展以支持</font></font><a href="/NVIDIA/cutlass/blob/main/include/cute/algorithm/prefetch.hpp"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">L2 预取</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">和</font></font><a href="/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp#L1337"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">TMA 存储+缩减</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">删除了一些 CUTLASS 2.x API 头文件的 C++11 要求。所有 CUTLASS 文件现在都需要 C++17。</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">修复以大大减少构建警告。</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">来自社区的更新和错误修复（谢谢！）</font></font></li>
</ul>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">最低要求：</font></font></p>
<ul dir="auto">
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">建筑：沃尔特</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">编译器：必须至少支持 C++17</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUDA 工具包版本：11.4</font></font></li>
</ul>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">从 CUTLASS 3.0 开始，CUTLASS 删除了对以下内容的支持：</font></font></p>
<ul dir="auto">
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Maxwell 和 Pascal GPU 架构</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">乌班图16.04</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUDA 10.2</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">C++ 语言版本低于 17。</font></font></li>
</ul>
<p dir="auto"><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">有关版本和更新的详细列表，</font><font style="vertical-align: inherit;">请参阅</font></font><a href="/NVIDIA/cutlass/blob/main/CHANGELOG.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">变更日志。</font></font></a><font style="vertical-align: inherit;"></font></strong></p>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">表现</font></font></h1><a id="user-content-performance" class="anchor" aria-label="永久链接：性能" href="#performance"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p align="center" dir="auto"><a target="_blank" rel="noopener noreferrer" href="/NVIDIA/cutlass/blob/main/media/images/cutlass-3.1-gemm-peak-performance.png"><img src="/NVIDIA/cutlass/raw/main/media/images/cutlass-3.1-gemm-peak-performance.png" style="max-width: 100%;"></a></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 原语非常高效。当用于构建设备范围的 GEMM 内核时，它们在标量 GEMM 计算方面表现出与 cuBLAS 相当的峰值性能。上图显示了</font></font><a href="https://www.nvidia.com/en-us/data-center/h100/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">NVIDIA H100</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">（NVIDIA Hopper 架构）、</font></font><a href="https://www.nvidia.com/en-us/data-center/l40/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">NVIDIA L40</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">（NVIDIA Ada 架构）、</font></font><a href="https://www.nvidia.com/en-us/data-center/a100/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">NVIDIA A100</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">（NVIDIA Ampere 架构）</font></font><br><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">
和</font></font><a href="https://www.nvidia.com/en-us/data-center/a40/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">NVIDIA A40</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">   （NVIDIA Ampere 架构）</font><font style="vertical-align: inherit;">上大矩阵维度下 CUTLASS 相对于 cuBLAS 的性能</font><font style="vertical-align: inherit;">。 CUTLASS 3.0 是使用</font></font><a href="https://developer.nvidia.com/cuda-downloads" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUDA 12.0 工具</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">包编译的。 Tensor Core 运算是使用 CUDA 的
</font></font><a href="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">mma</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">和
</font></font><a href="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">wgmma</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">指令实现的。</font></font></p>
<p align="center" dir="auto"><a target="_blank" rel="noopener noreferrer" href="/NVIDIA/cutlass/blob/main/media/images/cutlass-2.9-implicit-gemm-performance.png"><img src="/NVIDIA/cutlass/raw/main/media/images/cutlass-2.9-implicit-gemm-performance.png" style="max-width: 100%;"></a></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">当使用 CUTLASS 构建块构建设备范围的隐式 gemm（Fprop、Dgrad 和 Wgrad）内核时，在</font></font><a href="https://www.nvidia.com/en-us/data-center/a100/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">NVIDIA A100</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">上运行 Resnet-50 层时，CUTLASS 性能也与 cuDNN 相当
，如上图所示。 Tensor Core运算是使用CUDA的
</font></font><a href="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">mma指令</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">实现的。</font></font></p>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">兼容性</font></font></h1><a id="user-content-compatibility" class="anchor" aria-label="永久链接：兼容性" href="#compatibility"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 需要 C++17 主机编译器，并且在使用</font></font><a href="https://developer.nvidia.com/cuda-downloads" rel="nofollow"><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUDA 12.4 工具包</font></font></strong></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">构建时性能最佳。它还兼容 CUDA 11.4、CUDA 11.5、CUDA 11.6、CUDA 11.7、CUDA 11.8、CUDA 12.0、CUDA 12.1、CUDA 12.2.2、CUDA 12.3.1 和 CUDA 12.3.2。</font></font></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">操作系统</font></font></h2><a id="user-content-operating-systems" class="anchor" aria-label="永久链接：操作系统" href="#operating-systems"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">我们测试了以下环境。</font></font></p>
<table>
<thead>
<tr>
<th><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">操作系统</font></font></strong></th>
<th><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">编译器</font></font></strong></th>
</tr>
</thead>
<tbody>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">乌班图18.04</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">海湾合作委员会7.5.0</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">乌班图20.04</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">海湾合作委员会10.3.0</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">乌班图22.04</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">海湾合作委员会 11.2.0</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">乌班图22.04</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">铿锵10.0.0</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">乌班图22.04</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">铿锵14.0.6</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">乌班图22.04</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">铿锵17.0.6</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">视窗10.0</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Visual Studio 2019 v16.11.27</font></font></td>
</tr>
</tbody>
</table>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">注意：GCC 8.5.0 具有有关折叠表达式和重载运算符的已知回归。建议使用 GCC 7.5.0 或（首选）GCC &gt;= 9。</font></font></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">硬件</font></font></h2><a id="user-content-hardware" class="anchor" aria-label="永久链接：硬件" href="#hardware"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 在以下 NVIDIA GPU 上成功运行，预计在基于 Volta、Turing、Ampere、Ada 和 Hopper 架构的 NVIDIA GPU 上也能高效运行。</font></font></p>
<table>
<thead>
<tr>
<th><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">图形处理器</font></font></strong></th>
<th><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUDA计算能力</font></font></strong></th>
<th><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS-3 所需的最低 CUDA 工具包</font></font></strong></th>
</tr>
</thead>
<tbody>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">NVIDIA V100 张量核心 GPU</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">7.0</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">11.4</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">NVIDIA泰坦V</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">7.0</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">11.4</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">NVIDIA GeForce RTX 2080 TI、2080、2070</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">7.5</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">11.4</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">英伟达T4</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">7.5</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">11.4</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">NVIDIA A100 张量核心 GPU</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">8.0</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">11.4</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">英伟达A10</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">8.6</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">11.4</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">NVIDIA GeForce RTX 3090</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">8.6</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">11.4</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">NVIDIA GeForce RTX 4090</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">8.9</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">11.8</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">英伟达L40</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">8.9</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">11.8</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">NVIDIA H100 张量核心 GPU</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">9.0</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">11.8</font></font></td>
</tr>
</tbody>
</table>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">目标架构</font></font></h2><a id="user-content-target-architecture" class="anchor" aria-label="永久链接：目标架构" href="#target-architecture"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">一般来说，为一种目标架构生成的 PTX 代码可以在未来的架构上运行（即，它是向前兼容的）。然而，CUDA 12.0引入了“架构加速功能”的概念，其PTX没有前向兼容性保证。一些 Hopper PTX 指令属于此类架构加速功能，因此需要</font></font><code>sm_90a</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">目标架构（请注意附加的“a”）。有关此指令和其他架构加速指令的更多详细信息，请参阅</font></font><a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#feature-availability" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUDA 文档</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">目标架构信息通过 cmake 标志传递到 CUTLASS </font></font><code>CUTLASS_NVCC_ARCHS</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。为了最大限度地提高 Hopper GH100 的性能，用户需要构建 CUTLASS 作为</font></font><code>90a</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">目标架构。如果用户意外地使用 SM90 目标（注意缺少“a”）以及 CTK 12 或 11.8 构建了使用 SM90a 功能（例如 Hopper Tensor Core 指令）的内核，则内核预计会因运行时错误而失败。</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>cmake .. -DCUTLASS_NVCC_ARCHS="90a" 
</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="cmake .. -DCUTLASS_NVCC_ARCHS=&quot;90a&quot; " tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">请参阅</font></font><a href="/NVIDIA/cutlass/blob/main/media/docs/functionality.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">功能文档</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">，了解有关哪些内核需要哪些目标架构的详细信息。</font></font></p>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">文档</font></font></h1><a id="user-content-documentation" class="anchor" aria-label="永久链接：文档" href="#documentation"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">以下文档和随附的
</font></font><a href="https://nvidia.github.io/cutlass" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Doxygen 文档</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">中描述了 CUTLASS 。</font></font></p>
<ul dir="auto">
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/quickstart.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">快速入门指南</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">- 构建和运行 CUTLASS</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/functionality.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">功能</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">- 总结 CUTLASS 中可用的功能</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUDA 中的高效 GEMM</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;"> - 描述如何在 CUDA 中高效实现 GEMM 内核</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/cutlass_3x_design.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 3.x 设计</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">- 描述 CUTLASS 3.x 设计、其优点以及 CuTe 如何使我们能够编写更多可组合组件</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/gemm_api_3x.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">GEMM API 3.x</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;"> - 描述 CUTLASS 3.x GEMM 模型和 C++ 模板概念</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/gemm_api.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">GEMM API 2.x</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;"> - 描述 CUTLASS 2.x GEMM 模型和 C++ 模板概念</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/implicit_gemm_convolution.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">隐式 GEMM 卷积</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">- 描述 CUTLASS 中的 2-D 和 3-D 卷积</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/code_organization.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">代码组织</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">- 描述 CUTLASS 项目的组织和内容</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/terminology.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">术语</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">- 描述代码中使用的术语</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/programming_guidelines.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">编程指南</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">- 编写高效现代 CUDA C++ 的指南</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/fundamental_types.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">基本类型</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">- 描述 CUTLASS 中用于表示数值量和数组的基本 C++ 类</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/layout.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">布局</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">- 描述内存中矩阵和张量的布局</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/tile_iterator_concept.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Tile Iterators</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;"> - 描述了在内存中迭代矩阵块的 C++ 概念</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/profiler.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS Profiler</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;"> - 命令行驱动的分析应用程序</font></font></li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/utilities.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 实用程序</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">- 用于促进快速开发的附加模板</font></font></li>
</ul>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">资源</font></font></h1><a id="user-content-resources" class="anchor" aria-label="永久链接：资源" href="#resources"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"></font><a href="http://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">我们还在2018 年 GPU 技术大会</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">上的演讲中描述了高效 GEMM 的结构
</font><font style="vertical-align: inherit;">。</font></font></p>
<ul dir="auto">
<li><a href="https://www.nvidia.com/en-us/on-demand/session/gtcsiliconvalley2018-s8854/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS：CUDA 内所有级别和规模的密集线性代数的软件基元</font></font></a></li>
<li><a href="https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21745/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">开发 CUDA 内核将 Tensor Core 推向 NVIDIA A100 的绝对极限</font></font></a></li>
<li><a href="https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31883/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">在 CUTLASS 中使用张量核心加速卷积</font></font></a></li>
<li><a href="https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41996/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">通过增加 CUTLASS 中张量核心的利用率来加速后向数据梯度</font></font></a></li>
<li><a href="https://www.nvidia.com/en-us/on-demand/session/gtcfall22-a41131/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS：Python API、增强功能和 NVIDIA Hopper</font></font></a></li>
</ul>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">建造弯刀</font></font></h1><a id="user-content-building-cutlass" class="anchor" aria-label="永久链接：建造 CUTLASS" href="#building-cutlass"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 是一个仅包含头文件的模板库，不需要构建即可供其他项目使用。客户端应用程序应</font></font><code>include/</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">在其包含路径中</font><font style="vertical-align: inherit;">以 CUTLASS 目录为目标。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 单元测试、示例和实用程序可以使用 CMake 构建。</font></font><a href="/NVIDIA/cutlass/blob/main/media/docs/quickstart.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">快速入门指南</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">中给出了 CMake 的最低版本</font><font style="vertical-align: inherit;">。确保</font></font><code>CUDACXX</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">环境变量指向系统上安装的 CUDA 工具包中的 NVCC。</font></font></p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>$ <span class="pl-k">export</span> CUDACXX=<span class="pl-smi">${CUDA_INSTALL_PATH}</span>/bin/nvcc</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="$ export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">在 CUTLASS 项目中创建一个构建目录，然后运行 &ZeroWidthSpace;&ZeroWidthSpace;CMake。默认情况下，CUTLASS 将为 CUDA 架构版本 5.0、6.0、6.1、7.0、7.5、8.0、8.6、8.9 和 9.0 构建内核。为了减少编译时间，您可以通过更改 CMake 配置设置来指定构建 CUTLASS 的体系结构
</font></font><code>CUTLASS_NVCC_ARCHS</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。</font></font></p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>$ mkdir build <span class="pl-k">&amp;&amp;</span> <span class="pl-c1">cd</span> build

$ cmake .. -DCUTLASS_NVCC_ARCHS=80               <span class="pl-c"><span class="pl-c">#</span> compiles for NVIDIA's Ampere Architecture</span></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="$ mkdir build &amp;&amp; cd build

$ cmake .. -DCUTLASS_NVCC_ARCHS=80               # compiles for NVIDIA's Ampere Architecture" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">从</font></font><code>build/</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">目录中，通过使用 make 构建目标来编译并运行 CUTLASS 单元测试</font></font><code>test_unit</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">单元测试被组织为几个二进制文件，镜像 CUTLASS 的顶级命名空间，并且它们可以通过 make 的</font></font><code>-j</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">命令行参数并行执行。</font></font></p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>$ make test_unit -j
...
...
...
[----------] Global <span class="pl-c1">test</span> environment tear-down
[<span class="pl-k">==========</span>] 946 tests from 57 <span class="pl-c1">test</span> cases ran. (10812 ms total)
[  PASSED  ] 946 tests.</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="$ make test_unit -j
...
...
...
[----------] Global test environment tear-down
[==========] 946 tests from 57 test cases ran. (10812 ms total)
[  PASSED  ] 946 tests." tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">所有测试都应在支持的平台上通过，但测试的确切数量可能会随着时间的推移而变化。</font></font></p>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">项目结构</font></font></h1><a id="user-content-project-structure" class="anchor" aria-label="永久链接：项目结构" href="#project-structure"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 与实用程序、工具、示例和单元测试一起被安排为仅包含头文件的库。
 </font></font><a href="https://nvidia.github.io/cutlass" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Doxygen 文档</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">提供了 CUTLASS 项目中定义的文件、类和模板概念的完整列表。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">源代码组织的详细说明可以在
</font></font><a href="/NVIDIA/cutlass/blob/main/media/docs/code_organization.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 文档</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">中找到，但下面总结了几个主要组件。</font></font></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 模板库</font></font></h2><a id="user-content-cutlass-template-library" class="anchor" aria-label="永久链接：CUTLASS 模板库" href="#cutlass-template-library"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>include/                     # client applications should target this directory in their build's include paths

  cutlass/                   # CUDA Templates for Linear Algebra Subroutines and Solvers - headers only

    arch/                    # direct exposure of architecture features (including instruction-level GEMMs)

    conv/                    # code specialized for convolution

    epilogue/                # code specialized for the epilogue of gemm/convolution

    gemm/                    # code specialized for general matrix product computations

    layout/                  # layout definitions for matrices, tensors, and other mathematical objects in memory

    platform/                # CUDA-capable Standard Library components

    reduction/               # bandwidth-limited reduction kernels that do not fit the "gemm" model

    thread/                  # simt code that can be performed within a CUDA thread
    
    transform/               # code specialized for layout, type, and domain transformations

    *                        # core vocabulary types, containers, and basic numeric operations

  cute/                      # CuTe Layout, layout algebra, MMA/Copy atoms, tiled MMA/Copy

    algorithm/               # Definitions of core operations such as copy, gemm, and operations on cute::tuples

    arch/                    # Bare bones PTX wrapper structs for copy and math instructions

    atom/                    # Meta-information either link to or built from arch/ operators

      mma_atom.hpp           # cute::Mma_Atom and cute::TiledMma

      copy_atom.hpp          # cute::Copy_Atom and cute::TiledCopy

      *sm*.hpp               # Arch specific meta-information for copy and math operations

    *                        # Core library types such as Shape, Stride, Layout, Tensor, and associated operations

</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="include/                     # client applications should target this directory in their build's include paths

  cutlass/                   # CUDA Templates for Linear Algebra Subroutines and Solvers - headers only

    arch/                    # direct exposure of architecture features (including instruction-level GEMMs)

    conv/                    # code specialized for convolution

    epilogue/                # code specialized for the epilogue of gemm/convolution

    gemm/                    # code specialized for general matrix product computations

    layout/                  # layout definitions for matrices, tensors, and other mathematical objects in memory

    platform/                # CUDA-capable Standard Library components

    reduction/               # bandwidth-limited reduction kernels that do not fit the &quot;gemm&quot; model

    thread/                  # simt code that can be performed within a CUDA thread
    
    transform/               # code specialized for layout, type, and domain transformations

    *                        # core vocabulary types, containers, and basic numeric operations

  cute/                      # CuTe Layout, layout algebra, MMA/Copy atoms, tiled MMA/Copy

    algorithm/               # Definitions of core operations such as copy, gemm, and operations on cute::tuples

    arch/                    # Bare bones PTX wrapper structs for copy and math instructions

    atom/                    # Meta-information either link to or built from arch/ operators

      mma_atom.hpp           # cute::Mma_Atom and cute::TiledMma

      copy_atom.hpp          # cute::Copy_Atom and cute::TiledCopy

      *sm*.hpp               # Arch specific meta-information for copy and math operations

    *                        # Core library types such as Shape, Stride, Layout, Tensor, and associated operations
" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS SDK 示例</font></font></h3><a id="user-content-cutlass-sdk-examples" class="anchor" aria-label="永久链接：CUTLASS SDK 示例" href="#cutlass-sdk-examples"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><a href="/NVIDIA/cutlass/blob/main/examples"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS SDK示例</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">应用CUTLASS模板来实现基本计算。</font></font></p>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">工具</font></font></h3><a id="user-content-tools" class="anchor" aria-label="永久链接：工具" href="#tools"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>tools/
  library/                   # CUTLASS Instance Library - contains instantiations of all supported CUTLASS templates
    include/
      cutlass/
        library/

  profiler/                  # CUTLASS Profiler         - command-line utility for executing operations in the
                             #                            CUTLASS Library
  
  util/                      # CUTLASS Utilities        - contains numerous helper classes for
    include/                 #                            manging tensors in device memory, reference
      cutlass/               #                            implementations for GEMM, random initialization
        util/                #                            of tensors, and I/O.
</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="tools/
  library/                   # CUTLASS Instance Library - contains instantiations of all supported CUTLASS templates
    include/
      cutlass/
        library/

  profiler/                  # CUTLASS Profiler         - command-line utility for executing operations in the
                             #                            CUTLASS Library
  
  util/                      # CUTLASS Utilities        - contains numerous helper classes for
    include/                 #                            manging tensors in device memory, reference
      cutlass/               #                            implementations for GEMM, random initialization
        util/                #                            of tensors, and I/O." tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">测试</font></font></h3><a id="user-content-test" class="anchor" aria-label="永久链接：测试" href="#test"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">该</font></font><code>test/unit/</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">目录由使用 Google Test 实现的单元测试组成，演示了核心 API 组件的基本用法以及 CUTLASS GEMM 计算的完整测试。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"></font><a href="/NVIDIA/cutlass/blob/main/media/docs/quickstart.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">快速入门指南</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">中描述了构建和运行单元测试的说明</font><font style="vertical-align: inherit;">。</font></font></p>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">性能分析</font></font></h1><a id="user-content-performance-profiling" class="anchor" aria-label="永久链接：性能分析" href="#performance-profiling"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">该</font></font><code>tools/profiler/</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">目录包含用于启动每个 GEMM 内核的命令行实用程序。它可以构建如下：</font></font></p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>$ make cutlass_profiler -j16</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="$ make cutlass_profiler -j16" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">构建所有 GEMM 和卷积内核（</font><font style="vertical-align: inherit;">构建时间</font></font><em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">长）</font></font></em><font style="vertical-align: inherit;"></font></h2><a id="user-content-building-all-gemm-and-convolution-kernels-long-build-times" class="anchor" aria-label="永久链接：构建所有 GEMM 和卷积内核（构建时间长）" href="#building-all-gemm-and-convolution-kernels-long-build-times"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">默认情况下，只会为每种数据类型、数学指令和布局实例化一个图块大小。要实例化所有内容，请在从空目录运行 CMake 时设置以下环境变量</font></font><code>build/</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。请注意，这会导致</font></font><em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">数以万计</font></font></em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">的内核和较长的构建时间。这还会导致二进制大小过大，并且在某些平台上链接器无法构建库。因此，强烈建议仅生成内核的子集，如下面小节所示。</font></font></p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>$ cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_LIBRARY_KERNELS=all
...
$ make cutlass_profiler -j16</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="$ cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_LIBRARY_KERNELS=all
...
$ make cutlass_profiler -j16" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">构建 GEMM 和卷积内核的子集（</font></font><em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">减少</font></font></em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">构建时间）</font></font></h2><a id="user-content-building-a-subset-of-gemm-and-convolution-kernels-reduced-build-times" class="anchor" aria-label="永久链接：构建 GEMM 和卷积内核的子集（减少构建时间）" href="#building-a-subset-of-gemm-and-convolution-kernels-reduced-build-times"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">为了严格编译一个内核或一小组内核，可以使用带有通配符的逗号分隔的内核名称列表来减少内核集。以下示例展示了为 NVIDIA Ampere 和 Turing 架构构建一个或一个内核子集：</font></font></p>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">构建 Tensor Core GEMM 内核子集</font></font></h3><a id="user-content-building-a-subset-tensor-core-gemm-kernels" class="anchor" aria-label="永久链接：构建 Tensor Core GEMM 内核子集" href="#building-a-subset-tensor-core-gemm-kernels"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">要编译针对 NVIDIA Ampere 和 Turing 架构的具有 FP32 累积和 FP16 输入的 Tensor Core GEMM 内核子集，请使用以下 cmake 命令行：</font></font></p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>$ cmake .. -DCUTLASS_NVCC_ARCHS=<span class="pl-s"><span class="pl-pds">'</span>75;80<span class="pl-pds">'</span></span> -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s<span class="pl-k">*</span>gemm_f16_<span class="pl-k">*</span>_nt_align8
...
$ make cutlass_profiler -j16</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="$ cmake .. -DCUTLASS_NVCC_ARCHS='75;80' -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s*gemm_f16_*_nt_align8
...
$ make cutlass_profiler -j16" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">用于分析 Tensor Core GEMM 内核子集的示例命令行如下：</font></font></p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s<span class="pl-k">*</span>gemm_f16_<span class="pl-k">*</span>_nt_align8 --m=3456 --n=4096 --k=4096

...
=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_tensorop_s1688gemm_f16_256x128_32x2_nt_align8

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed
          cuBLAS: Passed

       Arguments: --gemm_kind=universal --m=3456 --n=4096 --k=4096 --A=f16:column --B=f16:row --C=f32:column --alpha=1  \
                  --beta=0 --split_k_slices=1 --batch_count=1 --op_class=tensorop --accum=f32 --cta_m=256 --cta_n=128  \
                  --cta_k=32 --stages=2 --warps_m=4 --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=8 --min_cc=75  \
                  --max_cc=1024

           Bytes: 118489088  bytes
           FLOPs: 115992428544  flops

         Runtime: 1.55948  ms
          Memory: 70.7616 GiB/s

            Math: 74378.8 GFLOP/s



=============================
...</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s*gemm_f16_*_nt_align8 --m=3456 --n=4096 --k=4096

...
=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_tensorop_s1688gemm_f16_256x128_32x2_nt_align8

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed
          cuBLAS: Passed

       Arguments: --gemm_kind=universal --m=3456 --n=4096 --k=4096 --A=f16:column --B=f16:row --C=f32:column --alpha=1  \
                  --beta=0 --split_k_slices=1 --batch_count=1 --op_class=tensorop --accum=f32 --cta_m=256 --cta_n=128  \
                  --cta_k=32 --stages=2 --warps_m=4 --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=8 --min_cc=75  \
                  --max_cc=1024

           Bytes: 118489088  bytes
           FLOPs: 115992428544  flops

         Runtime: 1.55948  ms
          Memory: 70.7616 GiB/s

            Math: 74378.8 GFLOP/s



=============================
..." tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">构建一个 CUDA Core GEMM 内核</font></font></h3><a id="user-content-building-one-cuda-core-gemm-kernel" class="anchor" aria-label="永久链接：构建一个 CUDA Core GEMM 内核" href="#building-one-cuda-core-gemm-kernel"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">要编译一个针对 NVIDIA Ampere 和 Turing 架构的 SGEMM 内核，请使用以下 cmake 命令行：</font></font></p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>$ cmake .. -DCUTLASS_NVCC_ARCHS=<span class="pl-s"><span class="pl-pds">'</span>75;80<span class="pl-pds">'</span></span> -DCUTLASS_LIBRARY_KERNELS=cutlass_simt_sgemm_128x128_8x2_nn_align1
...
$ make cutlass_profiler -j16</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="$ cmake .. -DCUTLASS_NVCC_ARCHS='75;80' -DCUTLASS_LIBRARY_KERNELS=cutlass_simt_sgemm_128x128_8x2_nn_align1
...
$ make cutlass_profiler -j16" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">用于分析单个 SGEMM CUDA 内核的示例命令行如下：</font></font></p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>$ ./tools/profiler/cutlass_profiler --kernels=sgemm --m=3456 --n=4096 --k=4096

=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_simt_sgemm_128x128_8x2_nn_align1

          Status: Success
    Verification: ON
     Disposition: Passed

          cuBLAS: Passed

       Arguments: --m=3456 --n=4096 --k=4096 --A=f32:column --B=f32:column --C=f32:column --alpha=1 --beta=0 --split_k_slices=1  \
                  --batch_count=1 --op_class=simt --accum=f32 --cta_m=128 --cta_n=128 --cta_k=8 --stages=2 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=1 --inst_n=1 --inst_k=1 --min_cc=50 --max_cc=1024

           Bytes: 180355072  bytes
           FLOPs: 115992428544  flops

         Runtime: 6.73655  ms
          Memory: 24.934 GiB/s

            Math: 17218.4 GFLOP/s

=============================</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="$ ./tools/profiler/cutlass_profiler --kernels=sgemm --m=3456 --n=4096 --k=4096

=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_simt_sgemm_128x128_8x2_nn_align1

          Status: Success
    Verification: ON
     Disposition: Passed

          cuBLAS: Passed

       Arguments: --m=3456 --n=4096 --k=4096 --A=f32:column --B=f32:column --C=f32:column --alpha=1 --beta=0 --split_k_slices=1  \
                  --batch_count=1 --op_class=simt --accum=f32 --cta_m=128 --cta_n=128 --cta_k=8 --stages=2 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=1 --inst_n=1 --inst_k=1 --min_cc=50 --max_cc=1024

           Bytes: 180355072  bytes
           FLOPs: 115992428544  flops

         Runtime: 6.73655  ms
          Memory: 24.934 GiB/s

            Math: 17218.4 GFLOP/s

=============================" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">构建 Tensor Core Convolution 内核的子集</font></font></h3><a id="user-content-building-a-subset-of-tensor-core-convolution-kernels" class="anchor" aria-label="永久链接：构建 Tensor Core 卷积内核的子集" href="#building-a-subset-of-tensor-core-convolution-kernels"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">要编译针对 NVIDIA Ampere 和 Turing 架构的具有 FP32 累积和 FP16 输入的前向传播 (fprop) 的 Tensor 核心卷积核子集，请使用以下 cmake 命令行：</font></font></p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>$ cmake .. -DCUTLASS_NVCC_ARCHS=<span class="pl-s"><span class="pl-pds">'</span>75;80<span class="pl-pds">'</span></span> -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s<span class="pl-k">*</span>fprop_optimized_f16
...
$ make cutlass_profiler -j16</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="$ cmake .. -DCUTLASS_NVCC_ARCHS='75;80' -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s*fprop_optimized_f16
...
$ make cutlass_profiler -j16" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">用于分析 Tensor Core 卷积核子集的示例命令行如下：</font></font></p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>$ ./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s<span class="pl-k">*</span>fprop_optimized_f16 --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3

...
=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: conv2d
       Operation: cutlass_tensorop_s16816fprop_optimized_f16_128x128_32x5_nhwc

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed

       Arguments: --conv_kind=fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --p=224 --q=224 --pad_h=1 --pad_w=1  \
                  --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1 --Activation=f16:nhwc --Filter=f16:nhwc --Output=f32:nhwc  \
                  --conv_mode=cross --iterator_algorithm=optimized --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1  \
                  --eq_gemm_provider=none --op_class=tensorop --accum=f32 --cta_m=128 --cta_n=128 --cta_k=32 --stages=5  \
                  --warps_m=2 --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=16 --min_cc=80 --max_cc=1024

           Bytes: 1130659840  bytes
           FLOPs: 118482796544  flops

         Runtime: 0.711496  ms
          Memory: 1479.99 GiB/s

            Math: 166526 GFLOP/s

=============================
...</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="$ ./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s*fprop_optimized_f16 --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3

...
=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: conv2d
       Operation: cutlass_tensorop_s16816fprop_optimized_f16_128x128_32x5_nhwc

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed

       Arguments: --conv_kind=fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --p=224 --q=224 --pad_h=1 --pad_w=1  \
                  --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1 --Activation=f16:nhwc --Filter=f16:nhwc --Output=f32:nhwc  \
                  --conv_mode=cross --iterator_algorithm=optimized --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1  \
                  --eq_gemm_provider=none --op_class=tensorop --accum=f32 --cta_m=128 --cta_n=128 --cta_k=32 --stages=5  \
                  --warps_m=2 --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=16 --min_cc=80 --max_cc=1024

           Bytes: 1130659840  bytes
           FLOPs: 118482796544  flops

         Runtime: 0.711496  ms
          Memory: 1479.99 GiB/s

            Math: 166526 GFLOP/s

=============================
..." tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">构建一个卷积 CUDA 内核</font></font></h3><a id="user-content-building-one-convolution-cuda-kernel" class="anchor" aria-label="永久链接：构建一个卷积 CUDA 内核" href="#building-one-convolution-cuda-kernel"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">要编译并运行一个针对 NVIDIA Ampere 和 Turing 架构的具有 F32 累积和 FP32 输入的前向传播 (fprop) 的 CUDA Core 卷积内核，请使用以下 cmake 命令行：</font></font></p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>$ cmake .. -DCUTLASS_NVCC_ARCHS=<span class="pl-s"><span class="pl-pds">'</span>75;80<span class="pl-pds">'</span></span> -DCUTLASS_LIBRARY_KERNELS=cutlass_simt_sfprop_optimized_128x128_8x2_nhwc
...
$ make cutlass_profiler -j16</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="$ cmake .. -DCUTLASS_NVCC_ARCHS='75;80' -DCUTLASS_LIBRARY_KERNELS=cutlass_simt_sfprop_optimized_128x128_8x2_nhwc
...
$ make cutlass_profiler -j16" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">用于分析一个 CUDA Core 卷积内核的示例命令行：</font></font></p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>$ ./tools/profiler/cutlass_profiler --kernels=cutlass_simt_sfprop_optimized_128x128_8x2_nhwc --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3


=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: conv2d
       Operation: cutlass_simt_sfprop_optimized_128x128_8x2_nhwc

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed

       Arguments: --conv_kind=fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --p=224 --q=224 --pad_h=1 --pad_w=1  \
                  --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1 --Activation=f32:nhwc --Filter=f32:nhwc --Output=f32:nhwc  \
                  --conv_mode=cross --iterator_algorithm=optimized --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1  \
                  --eq_gemm_provider=none --op_class=simt --accum=f32 --cta_m=128 --cta_n=128 --cta_k=8 --stages=2 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=1 --inst_n=1 --inst_k=1 --min_cc=50 --max_cc=1024

           Bytes: 2055798784  bytes
           FLOPs: 118482796544  flops

         Runtime: 7.34266  ms
          Memory: 260.752 GiB/s

            Math: 16136.2 GFLOP/s


=============================
</pre><div class="zeroclipboard-container">
  
  </div></div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">有关编译 CUTLASS 内核和 CUTLASS Profiler 的更多详细信息</font></font></h2><a id="user-content-more-details-on-compiling-cutlass-kernels-and-cutlass-profiler" class="anchor" aria-label="永久链接：有关编译 CUTLASS 内核和 CUTLASS Profiler 的更多详细信息" href="#more-details-on-compiling-cutlass-kernels-and-cutlass-profiler"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<ul dir="auto">
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">请点击以下链接，获取有关选择性编译 CUTLASS 内核的更多 CMake 示例：
</font></font><ul dir="auto">
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/quickstart.md#gemm-cmake-examples"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">GEMM CMake 示例</font></font></a></li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/quickstart.md#convolution-cmake-examples"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">隐式 GEMM 卷积 CMake 示例</font></font></a></li>
</ul>
</li>
<li><a href="/NVIDIA/cutlass/blob/main/media/docs/profiler.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">有关 CUTLASS Profiler 的更多详细信息请参见此处。</font></font></a></li>
</ul>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">关于</font></font></h1><a id="user-content-about" class="anchor" aria-label="永久链接：关于" href="#about"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"></font><a href="/NVIDIA/cutlass/blob/main/LICENSE.txt"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 是 NVIDIA 公司根据3 条“新”BSD 许可证</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">作为开源软件发布的
</font><font style="vertical-align: inherit;">。</font></font></p>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">贡献者</font></font></h1><a id="user-content-contributors" class="anchor" aria-label="永久链接：贡献者" href="#contributors"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">CUTLASS 开发者和贡献者的官方列表可在此处找到：</font></font><a href="/NVIDIA/cutlass/blob/main/CONTRIBUTORS.md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">贡献者</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。</font></font></p>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">版权</font></font></h1><a id="user-content-copyright" class="anchor" aria-label="永久链接：版权所有" href="#copyright"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">版权所有 (c) 2017 - 2024 NVIDIA 公司及附属公司。版权所有。 SPDX 许可证标识符：BSD-3 条款</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
</code></pre><div class="zeroclipboard-container">
  
  </div></div>
</article></div>
