# 实验 04：Register Tiling

## 从一个 `sum` 变成八个 `sum`

实验 03 已经让一个 block 合作把 A、B 小块装进 shared memory。不过，每个线程仍只算一个 `C[row,col]`，只有一个累加值 `sum`。

本实验设 `TM=8`：同一个线程负责**同一列的 8 个不同行**，并维护 `sums[0]` 到 `sums[7]`。例如第一个 block 中 `threadIdx=(2,3)` 负责 `C[3,2]`、`C[19,2]`、`C[35,2]`……直到 `C[115,2]`。这 8 个输出都有列号 2。

在固定的 `k`，这 8 个输出都需要同一个 `B[k,2]`。代码先从 shared memory 取一次 `b_tile[k][2]` 到线程自己的 `b`，再用它更新 8 个累加值：

```cpp
const float b = b_tile[k][local_col];
for (int i = 0; i < TM; ++i) {
  sums[i] += a_tile[local_row + i * TILE][k] * b;
}
```

`sums[8]` 是每个线程私有的短数组；编译器通常会把这些累加值放在寄存器中。`a_tile` 和 `b_tile` 仍是 **block 内共享**的数组。两种复用在一起工作：shared memory 供 block 的线程共同使用，寄存器中的多个 `sum` 让一个线程反复使用同一个 B 值。实际寄存器分配由编译器决定。

## 一个 block 现在负责多大区域

线程数仍是 `16×16=256`，但每个线程写最多 8 个输出，因此一个 block 覆盖 C 的 `128` 行、`16` 列。每轮沿 K 处理 16 个值：

| 数据 | 这一轮的大小 | 存放位置 |
| --- | ---: | --- |
| A 小块 | `128×16` | `a_tile`，shared memory |
| B 小块 | `16×16` | `b_tile`，shared memory |
| 每线程的 8 个累加值 | `8` | 线程私有，通常是寄存器 |

每个线程每轮负责填 8 个 A 位置和 1 个 B 位置；所有线程装完后同步，计算完后再同步。越界的 A/B 位置填零，最后只写有效的 C 元素。对于 `M=37`，上例的线程只会写前三个输出 `C[3,2]`、`C[19,2]`、`C[35,2]`。

与实验 03 相比，覆盖同一个 `128×16` 输出区域时，B 小块可以供这 8 组输出行共同使用；实验 03 需要 8 个 block 分别装入 B 小块。缓存可能已经帮实验 03 省下部分实际显存流量，因此不能只凭加载次数推断加速倍数。

## 运行与观察

在项目根目录运行：

```powershell
.\scripts\build-windows.ps1 -Lesson Register
```

Linux / WSL 可在本目录运行：

```bash
nvcc -O2 -std=c++17 -arch=sm_120 sgemm_register.cu -o sgemm_register
./sgemm_register
```

程序在**同一次运行**中比较 naive、实验 03 的 `16×16` tiled 基线和 `TM=8` register 版本。`37×53×29`、`129×257×65` 全量对照 CPU；大形状逐元素对照 naive GPU 输出，并抽查 CPU 参考值。预热后用 CUDA event 测 50 次 kernel 启动的平均时间，不包含分配和拷贝。

| 形状 | 本机三次运行的中位时间：tiled → register | 观察 |
| --- | --- | --- |
| `1×1024×1024` | `0.025 → 0.060 ms` | register 更慢；大部分输出行无效 |
| `64×1024×1024` | `0.080 → 0.061 ms` | register 略快 |
| `1024×1024×1024` | `1.076 → 0.402 ms` | register 约快 `2.68×` |

完整的三次结果和有效算力见[性能记录](../BENCHMARKS.md)。这组数字只说明当前实现和本机形状的表现；`TM=8` 不是所有矩阵形状的最佳选择。

## 可选练习：改变 `TM`

`TM` 越大，一个线程就有越多的 `sum`，一个 block 负责的 C 行数也越多：`16×TM`。这会增加线程内部对 B 的复用，也会增加每个线程的累加值、A 的 shared memory 小块，并减少同时可调度的 block 数量。因此速度不会一直随 `TM` 增长。

可以在 Windows 上直接改变编译时的值，默认值为 8：

```powershell
.\scripts\build-windows.ps1 -Lesson Register -ThreadTileM 16
```

Linux / WSL 可在 `nvcc` 命令中加入 `-DTHREAD_TILE_M=16`。本机 `1024³` 形状的交错测量结果如下；每项是三次程序运行的中位数：

| `TM` | 每个 block 负责的 C 行数 | register 时间 | 有效算力 |
| ---: | ---: | ---: | ---: |
| 4 | 64 | 0.478 ms | 4493.1 GFLOP/s |
| 8 | 128 | 0.401 ms | 5348.9 GFLOP/s |
| 16 | 256 | 0.541 ms | 3968.6 GFLOP/s |
| 32 | 512 | 0.916 ms | 2345.4 GFLOP/s |

这几个**自动编译版本**中 `TM=8` 最快。后来用 Nsight Compute 对 `TM=8/16` 采样，发现 `TM=16` 的 shared memory 加载指令多了 50%，MIO 队列等待增加。受控实验给 `TM=16` 强制展开 4 个 K 值后，机器指令从 `LDS.64` 变为 `LDS.128`，同形状时间从约 `0.541` 降到 `0.400 ms`，尽管实际 occupancy 更低；完整数据见 [TM 分析](TM_ANALYSIS.md)。

可以用下面的命令构建并运行强制展开版本；不传 `-KUnroll` 时由编译器自动决定。比较计时时，请使用相同的命令和矩阵形状分别测量两个版本，避免把上表与不同计时程序的数据直接相减。

```powershell
.\scripts\build-windows.ps1 -Lesson Register -ThreadTileM 16 -KUnroll 4
```

## 这节的练习

先在源码里找 `first_row`、`col` 和 `sums[TM]`。手算 `blockIdx=(0,0)`、`threadIdx=(4,1)` 写入的 8 个 C 坐标，然后指出它们在同一个 `k` 时共用哪个 B 元素。最后对照三个形状的计时，解释 `M=1` 为什么不适合这个 `128×16` 输出块。
