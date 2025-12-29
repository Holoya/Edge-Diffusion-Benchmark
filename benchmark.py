import torch
import time
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline, LCMScheduler

# 1. 设置硬件环境
# 默认用CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" 正在运行于: {device.upper()}")

# 设定提示词 (Prompt)
prompt = "A futuristic cyberpunk city with neon lights, highly detailed, 8k resolution, cinematic lighting"
#译文：一座未来风格的赛博朋克都市，霓虹闪烁，细节丰富，画面分辨率达 8K，配以电影级光影效果

# 实验 A: 原始 Stable Diffusion v1.5 (基准)

print("\n[1/3] 正在加载原始 SD v1.5 模型 (Baseline)...")
model_id = "runwayml/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# 预热
print("      正在预热 GPU...")
pipe("warmup", num_inference_steps=1)

print("      开始测试原始模型 (50 steps)...")
torch.cuda.reset_peak_memory_stats()
start_time = time.time()

# 生成图片
image_sd = pipe(prompt, num_inference_steps=50).images[0]

sd_time = time.time() - start_time
sd_memory = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
image_sd.save("result_baseline_sd.png")

print(f"      原始模型耗时: {sd_time:.2f} 秒 | 显存: {sd_memory:.2f} GB")

# 实验 B: 引入 LCM 加速 
print("\n[2/3] 正在加载 LCM 加速模块...")
# 核心点1：更换调度器为 LCM Scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
# 核心点2：加载 LCM LoRA 权重
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora()

print("      开始测试 LCM 加速模型 (4 steps)...")
torch.cuda.reset_peak_memory_stats()
start_time = time.time()


image_lcm = pipe(prompt, num_inference_steps=4, guidance_scale=1.0).images[0]

lcm_time = time.time() - start_time
lcm_memory = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
image_lcm.save("result_accelerated_lcm.png")

print(f"      LCM加速耗时: {lcm_time:.2f} 秒 | 显存: {lcm_memory:.2f} GB")


#绘制对比图表 

print("\n[3/3] 正在生成评测图表...")

# 计算加速比
speedup = sd_time / lcm_time
memory_saving = (1 - lcm_memory / sd_memory) * 100 if sd_memory > 0 else 0

# 设置画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 图1: 推理时间对比
models = ['Standard SD\n(50 Steps)', 'LCM Accelerated\n(4 Steps)']
times = [sd_time, lcm_time]
colors = ["#bf26f2", "#0099ff"] # 紫色和蓝色

bars = ax1.bar(models, times, color=colors, width=0.5)
ax1.set_title(f'Inference Latency (Speedup: {speedup:.1f}x)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Time (Seconds)')
# 在柱子上标数值
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}s', ha='center', va='bottom', fontsize=12)

# 图2: 显存占用对比
mems = [sd_memory, lcm_memory]
bars2 = ax2.bar(models, mems, color=["#bf26f2", "#0099ff"], width=0.5)
ax2.set_title('Peak VRAM Usage', fontsize=14, fontweight='bold')
ax2.set_ylabel('Memory (GB)')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}GB', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('benchmark_chart.png', dpi=300)
print(f"\n 实验完成！\n1. 图表已保存为 'benchmark_chart.png'\n2. 图片已保存为 'result_*.png'\n3. 加速比: {speedup:.1f}倍")
plt.show()