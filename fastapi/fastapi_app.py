import os
import json
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="ComfyUI FaceID Wrapper", version="1.0.0")

# -------- 配置 --------
COMFYUI_SERVER = os.getenv("COMFYUI_SERVER", "http://127.0.0.1:8188")
# ComfyUI 的输入/输出目录（跟你本机的 ComfyUI 配置保持一致）
COMFY_INPUT_DIR = Path(os.getenv("COMFY_INPUT_DIR", "./ComfyUI/input"))
COMFY_OUTPUT_DIR = Path(os.getenv("COMFY_OUTPUT_DIR", "./ComfyUI/output"))
# 工作流模板路径（使用你上传的 JSON）
WORKFLOW_JSON = Path(os.getenv("WORKFLOW_JSON", "./lora+faceid.json"))

COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
COMFY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- 辅助函数 --------
def load_workflow_template() -> Dict[str, Any]:
    if not WORKFLOW_JSON.exists():
        raise FileNotFoundError(f"Workflow JSON not found: {WORKFLOW_JSON}")
    with WORKFLOW_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)

def patch_workflow(
    wf: Dict[str, Any],
    image_filename: str,
    prompt: Optional[str],
    negative_prompt: Optional[str],
    seed: Optional[int],
    steps: Optional[int],
    cfg: Optional[float],
    width: Optional[int],
    height: Optional[int],
) -> Dict[str, Any]:
    """
    根据入参修改工作流节点：
    - 7: LoadImage.image
    - 11/12: CLIPTextEncode.text
    - 13: KSampler 采样参数
    - 34: EmptyLatentImage.width/height
    """
    def set_in(node_id: str, key: str, value):
        if node_id not in wf:
            raise KeyError(f"Node {node_id} not in workflow")
        wf[node_id]["inputs"][key] = value

    # 1) 图像（LoadImage, node 7）
    if image_filename:
        set_in("7", "image", image_filename)

    # 2) 正/负面提示词（CLIPTextEncode，11/12）
    if prompt is not None:
        set_in("11", "text", prompt)
    if negative_prompt is not None:
        set_in("12", "text", negative_prompt)

    # 3) 采样器参数（KSampler，13）
    if seed is not None:
        set_in("13", "seed", seed)
    if steps is not None:
        set_in("13", "steps", steps)
    if cfg is not None:
        set_in("13", "cfg", cfg)

    # 4) 尺寸（EmptyLatentImage，34）
    if width is not None:
        set_in("34", "width", width)
    if height is not None:
        set_in("34", "height", height)

    return wf

async def submit_prompt(wf: Dict[str, Any]) -> str:
    """
    调用 ComfyUI /prompt，返回 prompt_id
    """
    payload = {"prompt": wf}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{COMFYUI_SERVER}/prompt", json=payload)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"ComfyUI /prompt error: {r.text}")
        data = r.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            # 老版本返回 {'node_errors':..., 'number':...}；做兼容
            prompt_id = str(uuid.uuid4())
        return prompt_id

async def wait_for_result(prompt_id: str, poll_interval: float = 0.8, timeout_sec: int = 300) -> Dict[str, Any]:
    """
    轮询 /history/{prompt_id}，直到有图像输出或超时
    返回 history JSON（包含输出文件名）
    """
    deadline = time.time() + timeout_sec
    async with httpx.AsyncClient(timeout=30) as client:
        while time.time() < deadline:
            r = await client.get(f"{COMFYUI_SERVER}/history/{prompt_id}")
            if r.status_code == 200:
                hist = r.json()
                # 结构示例：{prompt_id: {"outputs": {"51": {"images": [{"filename": "...", "subfolder": "", "type": "output"}]}}}}
                if prompt_id in hist:
                    outputs = hist[prompt_id].get("outputs", {})
                    # 寻找包含 images 的任意节点输出
                    for node_id, out in outputs.items():
                        if "images" in out and out["images"]:
                            return hist
            await asyncio_sleep(poll_interval)
        raise HTTPException(status_code=504, detail="Timed out waiting for ComfyUI result")

# 兼容没有显式导入 asyncio 的睡眠
import asyncio
async def asyncio_sleep(s: float):
    await asyncio.sleep(s)

def save_upload_to_input_dir(up: UploadFile) -> str:
    """
    把上传的图片保存到 ComfyUI input 目录，返回保存的文件名（不带路径）
    """
    suffix = Path(up.filename).suffix.lower() or ".png"
    safe_name = f"face_{uuid.uuid4().hex}{suffix}"
    target = COMFY_INPUT_DIR / safe_name
    with target.open("wb") as f:
        f.write(up.file.read())
    return safe_name

# -------- API --------
@app.post("/generate")
async def generate(
    face: UploadFile = File(..., description="Face image for IP-Adapter FaceID"),
    prompt: Optional[str] = Form(None, description="Positive prompt to override node 11"),
    negative_prompt: Optional[str] = Form(None, description="Negative prompt to override node 12"),
    seed: Optional[int] = Form(None),
    steps: Optional[int] = Form(None),
    cfg: Optional[float] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
):
    """
    封装工作流一次推理：上传人脸 + 可选超参，返回输出文件列表
    """
    # 1) 保存上传图片到 ComfyUI/input
    image_filename = save_upload_to_input_dir(face)

    # 2) 载入并打补丁
    wf = load_workflow_template()
    wf = patch_workflow(
        wf, image_filename, prompt, negative_prompt, seed, steps, cfg,
         width, height
    )

    # 3) 提交到 ComfyUI
    prompt_id = await submit_prompt(wf)

    # 4) 轮询结果
    hist = await wait_for_result(prompt_id)

    # 5) 提取输出图片（相对路径）
    outputs = hist[prompt_id].get("outputs", {})
    images = []
    for node_id, out in outputs.items():
        for im in out.get("images", []):
            # 通常 type: "output"，subfolder 可能为空或子目录名
            images.append({
                "node": node_id,
                "filename": im.get("filename"),
                "subfolder": im.get("subfolder", ""),
                "type": im.get("type", ""),
                "path_hint": str(COMFY_OUTPUT_DIR / im.get("filename", "")),
            })

    return JSONResponse({"prompt_id": prompt_id, "images": images})

@app.get("/health")
async def health():
    # 简单探活：ping /queue 参数
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(f"{COMFYUI_SERVER}/queue")
            ok = (r.status_code == 200)
        except Exception:
            ok = False
    return {"ok": ok}
