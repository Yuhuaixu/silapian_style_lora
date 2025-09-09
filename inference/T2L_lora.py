import torch

from diffusers import StableDiffusionXLPipeline,AutoencoderKL,DPMSolverMultistepScheduler


#模型路径，按需更换为自己的路径

base_model_dir = "/root/autodl-tmp/model/sdxl_base/snapshots/base"

vae_dir = "/root/autodl-tmp/VAE/snapshots/vae"

lora_checkpoints_dir = "/root/autodl-tmp/Lora/silapian_style/checkpoint-1400"

#推理参数

prompt = "silapian_style,a girl,white hair,portrait,masterpiece"

negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

guidance_scale = 3

num_inference_step = 35

generator = torch.Generator("cuda").manual_seed(0)

#加载模型

vae = AutoencoderKL.from_pretrained(vae_dir,torch_dtype = torch.float16,variant="fp16").to("cuda")

base = StableDiffusionXLPipeline.from_pretrained(
    base_model_dir,vae = vae,torch_dtype=torch.float16,
    variant="fp16",
    ).to("cuda")

base.scheduler = DPMSolverMultistepScheduler.from_config(base.scheduler.config)

base.load_lora_weights(lora_checkpoints_dir,adapter_name="silapian")

base.set_adapters(adapter_names="silapian",adapter_weights=1)


#推理生成

image = base(prompt=prompt,negative_prompt=negative_prompt,
             guidance_scale=guidance_scale,
             num_inference_steps = num_inference_step,
             generator = generator).images[0]
image