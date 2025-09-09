import os
import warnings
import torch
import cv2
import gradio as gr
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.utils import load_image
# from transformers import CLIPVisionModelWithProjection  # unused

try:
    from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL
    HAS_IP_ADAPTER = True
except ImportError:
    HAS_IP_ADAPTER = False
    print("警告: 未安装 ip_adapter，请关闭 Face ID 功能或按需安装")


class SDXLImg2ImgDemo:
    def __init__(self, suppress_warnings: bool = False):
        # 设备 & 精度
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if suppress_warnings:
            # 可选：抑制已知的非致命告警（更适合在 Notebook/生产中减少噪音）
            warnings.filterwarnings(
                "ignore", message=r"Already found a `peft_config` attribute"
            )
            warnings.filterwarnings(
                "ignore", message=r"Adapter cannot be set when the model is merged"
            )
            warnings.filterwarnings(
                "ignore", category=FutureWarning, module=r"diffusers\.models\.lora"
            )

        # ===== 模型路径（按需修改到你本地的路径） =====
        self.base_model_dir = "/root/autodl-tmp/model/sdxl_base/snapshots/base"
        self.ip_adapter_ckpt = "/root/autodl-tmp/ComfyUI/models/ipadapter/ip-adapter-faceid_sdxl.bin"
        self.lora_checkpoints_dir = "/root/autodl-tmp/Lora/silapian_style/checkpoint-1400"

        # ===== 运行时开关 =====
        self.lora_adapter_name = "style_lora"
        self.lora_loaded = False
        self.lora_fused = False  # 我们默认不使用 fuse；若你强制 fuse，会设置为 True

        print("正在加载 SDXL 模型…")
        self.load_models()
        print("模型加载完成！")

    # --------------------- 加载模块 ---------------------
    def load_models(self):
        # 1) VAE

        # 2) SDXL pipeline
        print("加载 SDXL 基础模型…")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.base_model_dir,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if torch.cuda.is_available() else None,
        ).to(self.device)

        # 3) 调度器
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        # 4) Face ID 组件（可选）
        self.ip_adapter = None
        self.face_app = None
        if HAS_IP_ADAPTER and os.path.exists(self.ip_adapter_ckpt):
            try:
                from insightface.app import FaceAnalysis

                print("初始化人脸分析模型…")
                # insightface: GPU 用 ctx_id=0；CPU 用 -1
                ctx_id = 0 if self.device == "cuda" else -1
                self.face_app = FaceAnalysis(
                    name="buffalo_l",
                    root="/root/autodl-tmp/ComfyUI/models/insightface",
                )
                self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

                
                print("Face ID 功能已启用")
            except Exception as e:
                print(f"Face ID 初始化失败: {e}")
                self.ip_adapter = None
                self.face_app = None
        else:
            print("未启用 Face ID（缺少依赖或权重文件）")

    # --------------------- LoRA 管理 ---------------------
    def load_lora(self, lora_weight: float = 0.7):
        """按需加载/设置 LoRA。默认不 fuse，避免“合并后无法再设置适配器”的告警。"""
        try:
            # 如果之前做过 fuse，先尝试 unfuse（不同版本是否存在 API 需 try/except）
            if self.lora_fused:
                try:
                    self.pipe.unfuse_lora()
                    self.lora_fused = False
                except Exception:
                    pass

            if not self.lora_loaded:
                # 仅首次加载一次适配器，命名以便后续 set_adapters 调整权重
                self.pipe.load_lora_weights(
                    self.lora_checkpoints_dir,
                    adapter_name=self.lora_adapter_name,
                )
                self.lora_loaded = True

            # 设置当前 LoRA 权重（不合并）
            try:
                # 新版 diffusers 支持列表形式
                self.pipe.set_adapters([self.lora_adapter_name], [float(lora_weight)])
            except Exception:
                # 兼容旧版 API
                self.pipe.set_adapters(self.lora_adapter_name, float(lora_weight))

            return f"LoRA 就绪（{os.path.basename(self.lora_checkpoints_dir)}，权重={lora_weight}）"
        except Exception as e:
            return f"LoRA 加载/设置失败: {str(e)}"

    def fuse_lora_once(self, lora_scale: float = 0.7):
        """如你确定不再切换适配器，可选择合并；合并后不要再尝试 set_adapter。"""
        if not self.lora_loaded:
            self.load_lora(lora_scale)
        try:
            self.pipe.fuse_lora(lora_scale=float(lora_scale))
            self.lora_fused = True
            return f"LoRA 已合并 (scale={lora_scale})"
        except Exception as e:
            return f"LoRA 合并失败: {e}"

    # --------------------- FaceID 特征 ---------------------
    def extract_face_embedding(self, face_image):
        if face_image is None or self.face_app is None:
            return None
        try:
            # 转 PIL -> OpenCV (BGR)
            if isinstance(face_image, Image.Image):
                face_image_cv = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
            else:
                face_image_cv = face_image

            faces = self.face_app.get(face_image_cv)
            if len(faces) == 0:
                return None

            faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
            return faceid_embeds
        except Exception as e:
            print(f"人脸提取失败: {e}")
            return None

    # --------------------- 生成 ---------------------
    def generate_image(
        self,
        prompt,
        negative_prompt="",
        face_image=None,
        lora_weight=0.7,
        face_id_weight=0.8,
        guidance_scale=7.5,
        num_inference_steps=20,
        seed=42,
    ):
        try:
            # 规范化 seed（Gradio Number 可能是 float）
            try:
                seed = int(seed)
            except Exception:
                seed = -1

            generator = None
            if seed != -1:
                generator = torch.Generator(self.device).manual_seed(seed)

            status_msgs = []

            # LoRA：仅设置/加载，不执行 fuse（避免与 IP-Adapter 冲突 + 告警）
            lora_status = self.load_lora(lora_weight)
            status_msgs.append(lora_status)

            # Face ID 分支
            if face_image is not None and os.path.exists(self.ip_adapter_ckpt):
                self.ip_adapter = IPAdapterFaceIDXL(self.pipe, self.ip_adapter_ckpt, self.device)
                face_embeds = self.extract_face_embedding(face_image)
                if face_embeds is not None:
                    images = self.ip_adapter.generate(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=float(guidance_scale),
                        num_inference_steps=int(num_inference_steps),
                        seed=seed,
                        num_samples=1,
                        faceid_embeds=face_embeds,
                        scale=float(face_id_weight),
                    )

                    result_image = images[0] if isinstance(images, list) else images
                    if isinstance(result_image, np.ndarray):
                        result_image = Image.fromarray(result_image)
                    status_msgs.append("使用 Face ID 生成完成")
                    return result_image, " | ".join(status_msgs)
                else:
                    status_msgs.append("Face ID 提取失败，回退到普通模式")

        except Exception as e:
            import traceback

            traceback.print_exc()
            return None, f"生成图像时出错: {str(e)}"


# --------------------- Gradio ---------------------

def create_gradio_interface():
    demo_instance = SDXLImg2ImgDemo(suppress_warnings=False)

    def generate_wrapper(prompt, negative_prompt, face_img, lora_weight, face_id_weight, guidance_scale, num_inference_steps, seed):
        return demo_instance.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            face_image=face_img,
            lora_weight=lora_weight,
            face_id_weight=face_id_weight,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

    interface = gr.Interface(
        fn=generate_wrapper,
        inputs=[
            gr.Textbox(label="正向提示词", lines=3, value="silapian_style,a girl,white hair,portrait,masterpiece", placeholder="输入描述生成图像的提示词…"),
            gr.Textbox(label="负向提示词", lines=2, value="blurry, ugly, deformed, grain, low-res, (worst quality:1.4)", placeholder="输入不希望出现的内容…"),
            gr.Image(label="参考人脸图像 (Face ID)", type="pil"),
            gr.Slider(label="LoRA 权重", minimum=0.0, maximum=2.0, value=1.0, step=0.1),
            gr.Slider(label="Face ID 权重", minimum=0.0, maximum=2.0, value=0.8, step=0.1),
            gr.Slider(label="引导尺度", minimum=1.0, maximum=20.0, value=7.5, step=0.5),
            gr.Slider(label="推理步数", minimum=10, maximum=50, value=20, step=1),
            gr.Number(label="随机种子 (-1 为随机)", value=42),
        ],
        outputs=[
            gr.Image(label="生成图像"),
            gr.Textbox(label="状态信息"),
        ],
        title="SDXL 文生图 + LoRA + IP-Adapter Face ID",
        description=(
            "集成 LoRA 和 IP-Adapter Face ID 的图像生成工具\n\n"
            "使用说明：\n"
            "1. 输入描述性的提示词\n"
            "2. (可选) 上传人脸参考图像启用 Face ID\n"
            "3. (可选) 调整 LoRA 权重\n"
            "4. 调整参数后点击提交生成图像"
        ),
    )

    return interface


if __name__ == "__main__":
    try:
        demo = create_gradio_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    except Exception as e:
        print(f"启动失败: {e}")
