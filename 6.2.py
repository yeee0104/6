import os
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
import tkinter as tk
from tkinter import messagebox

# 禁用 Hugging Face 符号链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 定义生成图片的函数
def generate_image():
    input_text = input_entry.get()  # 获取输入框的内容
    if not input_text or not input_text.strip():
        messagebox.showerror("错误", "输入不能为空，请输入有效的句子！")
        return

    print("正在翻译中文句子...")
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
    translation = translator(input_text)[0]['translation_text']
    print(f"翻译结果: {translation}")

    # 第二步：使用翻译后的英文句子生成图片
    print("正在加载 Stable Diffusion 模型...")
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

    # 检查是否有 GPU，否则切换为 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # 移除 float16 的设置，确保在 CPU 上使用 float32
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    pipe = pipe.to(device)

    # 开始生成图片
    print("开始生成图片...")
    try:
        image = pipe(translation).images[0]
        print("图片生成成功！")

        # 获取当前脚本所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, "generated_image.png")
        
        # 保存生成的图片
        image.save(save_path)
        print(f"图片已生成并保存为 '{save_path}'。")
        messagebox.showinfo("成功", f"图片已生成并保存为\n{save_path}")
    except Exception as e:
        print("图片生成失败！")
        print(f"错误信息：{e}")
        messagebox.showerror("错误", f"图片生成失败！\n错误信息：{e}")

# 创建 Tkinter 窗口
root = tk.Tk()
root.title("图片生成器")

# 创建标签和输入框
tk.Label(root, text="请输入一个中文句子：").pack(pady=10)
input_entry = tk.Entry(root, width=50)
input_entry.pack(pady=10)

# 创建生成按钮
generate_button = tk.Button(root, text="生成图片", command=generate_image)
generate_button.pack(pady=20)

# 启动 Tkinter 主循环
root.mainloop()