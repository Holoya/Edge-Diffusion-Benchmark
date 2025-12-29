项目基于 Stable Diffusion v1.5 对比了原始模型与 LCM-LoRA 加速技术的性能。
文件说明
- benchmark.py: 核心测试代码，直接运行即可
- benchmark_chart.png: 自动生成的性能对比图
- result_*.png: 生成的效果对比图（2张）
环境依赖
- torch: 2.9.1+cu128
- diffusers: 0.36.0
- transformers: 4.57.3
- accelerate: 1.12.0
- matplotlib: 3.10.8
