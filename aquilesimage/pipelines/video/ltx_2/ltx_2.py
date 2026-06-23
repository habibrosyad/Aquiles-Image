from aquilesimage.utils.utils_video import get_path_file_video_model, file_exists, download_ltx_2, download_ltx_2_3
from typing import Literal
try:
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_pipelines.utils.args import ImageConditioningInput
except ImportError as e:
    print("Error importing components for LTX-2")
    pass
import torch
import gc

class LTX_2_Pipeline:
    FRAME_RATE = 25.0

    def __init__(self, model_name: Literal["ltx-2", "ltx-2.3"] = "ltx-2"):
        self.pipeline: TI2VidTwoStagesPipeline | None = None
        self.model_name = model_name
        self.verify_model()

    @staticmethod
    def _compute_num_frames(seconds: str) -> int:
        target = int(float(seconds) * LTX_2_Pipeline.FRAME_RATE)
        k = max(round((target - 1) / 8), 1)
        return 8 * k + 1

    @staticmethod
    def _align_resolution(h: int, w: int, alignment: int = 32) -> tuple[int, int]:
        return (h // alignment) * alignment, (w // alignment) * alignment

    @staticmethod
    def _normalize_images(image, num_frames: int) -> list:
        if image is None:
            return []
        if isinstance(image, list):
            if len(image) == 0:
                return []
            if isinstance(image[0], (list, tuple)):
                return [ImageConditioningInput(img, pos, str_) for img, pos, str_ in image]
            if len(image) == 1:
                return [ImageConditioningInput(image[0], 0, 1.0)]
            n = len(image)
            positions = [int(i * (num_frames - 1) / (n - 1)) for i in range(n)]
            return [ImageConditioningInput(img, pos, 1.0) for img, pos in zip(image, positions)]
        return [ImageConditioningInput(image, 0, 1.0)]

    def start(self):
        data_dir = get_path_file_video_model(self.model_name)

        if self.model_name == "ltx-2":
            checkpoint_path = f"{data_dir}/ltx-2-19b-dev.safetensors"
            distilled_lora = f"{data_dir}/ltx-2-19b-distilled-lora-384.safetensors"
            spatial_upsampler_path = f"{data_dir}/ltx-2-spatial-upscaler-x2-1.0.safetensors"
        elif self.model_name == "ltx-2.3":
            checkpoint_path = f"{data_dir}/ltx-2.3-22b-dev.safetensors"
            distilled_lora = f"{data_dir}/ltx-2.3-22b-distilled-lora-384.safetensors"
            spatial_upsampler_path = f"{data_dir}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        else:
            raise ValueError("Model not available")

        with torch.no_grad():
            self.pipeline = TI2VidTwoStagesPipeline(
                checkpoint_path=checkpoint_path,
                gemma_root=f"{data_dir}/gemma", 
                loras=[], 
                distilled_lora=[
                    LoraPathStrengthAndSDOps(
                        path=distilled_lora, 
                        strength=0.6, 
                        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP
                    )
                ], 
                spatial_upsampler_path=spatial_upsampler_path
            )

    def generate(self, seed: int, prompt: str, save_result_path: str, negative_prompt: str, image=None, seconds=None, height=1088, width=1920, num_inference_steps=40):
        try:
            import os
            output_dir = os.path.dirname(save_result_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with torch.no_grad():
                height, width = self._align_resolution(height, width, 32)
                num_frames = self._compute_num_frames(seconds) if seconds is not None else 193
                tiling_config = TilingConfig.default()
                video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

                images_list = self._normalize_images(image, num_frames)

                video, audio = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=self.FRAME_RATE,
                    num_inference_steps=num_inference_steps,
                    images=images_list,
                    video_guider_params=MultiModalGuiderParams(
                        cfg_scale=3.0,
                        stg_scale=1.0,
                        rescale_scale=0.7,
                        modality_scale=3.0,
                        skip_step=0,
                        stg_blocks=[29],
                    ),
                    audio_guider_params=MultiModalGuiderParams(
                        cfg_scale=7.0,
                        stg_scale=1.0,
                        rescale_scale=0.7,
                        modality_scale=3.0,
                        skip_step=0,
                        stg_blocks=[29],
                    ),
                    enhance_prompt=False,
                    tiling_config=tiling_config,
                )

                encode_video(
                    video=video,
                    fps=self.FRAME_RATE,
                    audio=audio,
                    output_path=save_result_path,
                    video_chunks_number=video_chunks_number,
                )

            print(f"Saved video in... {save_result_path}")

        except Exception as e:
            print(f"X Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.ipc_collect()
            gc.collect()

    def verify_model(self):
        model_path = get_path_file_video_model(self.model_name)

        if self.model_name == "ltx-2":
            if (file_exists(f"{model_path}/gemma/model-00004-of-00005.safetensors") and 
                file_exists(f"{model_path}/ltx-2-19b-dev.safetensors") and 
                file_exists(f"{model_path}/ltx-2-spatial-upscaler-x2-1.0.safetensors") and 
                file_exists(f"{model_path}/ltx-2-19b-distilled-lora-384.safetensors")):
                pass
            else:
                download_ltx_2()
        elif self.model_name == "ltx-2.3":
            if (file_exists(f"{model_path}/gemma/model-00004-of-00005.safetensors") and 
                file_exists(f"{model_path}/ltx-2.3-22b-dev.safetensors") and 
                file_exists(f"{model_path}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors") and 
                file_exists(f"{model_path}/ltx-2.3-22b-distilled-lora-384.safetensors")):
                pass
            else:
                download_ltx_2_3()