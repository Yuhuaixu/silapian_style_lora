import torch
import cv2
from diffusers import StableDiffusionXLPipeline,AutoencoderKL,DPMSolverMultistepScheduler
from insightface.app import FaceAnalysis
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL

#模型路径
base_model_dir = "/root/autodl-tmp/model/sdxl_base/snapshots/base"
vae_dir = "/root/autodl-tmp/VAE/snapshots/vae"
lora_checkpoints_dir = "/root/autodl-tmp/Lora/silapian_style/checkpoint-1400"
ip_adapter_dir = "/root/autodl-tmp/ComfyUI/models/ipadapter/ip-adapter-faceid_sdxl.bin"


#参数设置
prompt = "silapian_style,a girl,white background,portrait,masterpiece"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
guidance_scale = 3
num_inference_step = 20
generator = torch.Generator("cuda").manual_seed(0)
face_scale = 1.0
device = "cuda"

#提取人脸信息

face_app = FaceAnalysis(name="buffalo_1",root="/root/autodl-tmp/ComfyUI/models/insightface")

face_app.prepare(ctx_id=0, det_size=(640, 640))

faceid_image = cv2.imread("/root/autodl-tmp/dataset/train/prior_photo_woman_50/1.png")

face = face_app.get(faceid_image)

faceid_embeds = torch.from_numpy(face[0].normed_embedding).unsqueeze(0)


#加载模型
vae = AutoencoderKL.from_pretrained(vae_dir,torch_dtype = torch.float16,variant="fp16").to("cuda")

base = StableDiffusionXLPipeline.from_pretrained(
    base_model_dir,vae = vae,torch_dtype=torch.float16,
    variant="fp16",
    ).to("cuda")

base.scheduler = DPMSolverMultistepScheduler.from_config(base.scheduler.config)


base.load_lora_weights(lora_checkpoints_dir)
base.fuse_lora(lora_scale=1)


pipe = IPAdapterFaceIDXL(base,ip_adapter_dir,device)

#图片生成

image = pipe.generate(prompt=prompt,negative_prompt=negative_prompt,
             guidance_scale=guidance_scale,
             num_inference_steps = num_inference_step,
             seed=0,num_samples=1,scale=face_scale,
             faceid_embeds=faceid_embeds)

image[0]