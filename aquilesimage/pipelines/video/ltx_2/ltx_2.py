from aquilesimage.utils.utils_video import get_path_file_video_model, file_exists, download_ltx_2, download_ltx_2_3
from typing import Literal
try:
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_core.model.video_vae import TilingConfig, SpatialTilingConfig, TemporalTilingConfig, get_video_chunks_number
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

    def generate(
        self,
        seed: int,
        prompt: str,
        save_result_path: str,
        negative_prompt: str = "No deformities",
        image=None,
        seconds=None,
        height: int = 1088,
        width: int = 1920,
        num_inference_steps: int = 40,
        video_cfg_scale: float = 3.0,
        video_stg_scale: float = 1.0,
        video_rescale_scale: float = 0.7,
        a2v_guidance_scale: float = 3.0,
        video_skip_step: int = 0,
        video_stg_blocks: list[int] | None = None,
        audio_cfg_scale: float = 7.0,
        audio_stg_scale: float = 1.0,
        audio_rescale_scale: float = 0.7,
        v2a_guidance_scale: float = 3.0,
        audio_skip_step: int = 0,
        audio_stg_blocks: list[int] | None = None,
        frame_rate: float = 25.0,
        enhance_prompt: bool = False,
        image_strength: float = 1.0,
        image_frame_idx: int = 0,
        stage_1_sigmas: list[float] | None = None,
        stage_2_sigmas: list[float] | None = None,
        max_batch_size: int = 1,
        spatial_tile_size: int | None = None,
        spatial_tile_overlap: int = 64,
        temporal_tile_size: int | None = None,
        temporal_tile_overlap: int = 24,
    ):
        try:
            import os
            output_dir = os.path.dirname(save_result_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            height, width = self._align_resolution(height, width, 32)
            num_frames = self._compute_num_frames(seconds) if seconds is not None else 193

            spatial = (
                SpatialTilingConfig(tile_size_in_pixels=spatial_tile_size, tile_overlap_in_pixels=spatial_tile_overlap)
                if spatial_tile_size is not None else None
            )
            temporal = (
                TemporalTilingConfig(tile_size_in_frames=temporal_tile_size, tile_overlap_in_frames=temporal_tile_overlap)
                if temporal_tile_size is not None else None
            )
            tiling_config = TilingConfig(spatial_config=spatial, temporal_config=temporal) if spatial or temporal else TilingConfig.default()
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

            with torch.no_grad():
                pipeline_kwargs = dict(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    num_inference_steps=num_inference_steps,
                    images=[ImageConditioningInput(image, image_frame_idx, image_strength)] if image is not None else [],
                    video_guider_params=MultiModalGuiderParams(
                        cfg_scale=video_cfg_scale,
                        stg_scale=video_stg_scale,
                        rescale_scale=video_rescale_scale,
                        modality_scale=a2v_guidance_scale,
                        skip_step=video_skip_step,
                        stg_blocks=video_stg_blocks if video_stg_blocks is not None else [29],
                    ),
                    audio_guider_params=MultiModalGuiderParams(
                        cfg_scale=audio_cfg_scale,
                        stg_scale=audio_stg_scale,
                        rescale_scale=audio_rescale_scale,
                        modality_scale=v2a_guidance_scale,
                        skip_step=audio_skip_step,
                        stg_blocks=audio_stg_blocks if audio_stg_blocks is not None else [29],
                    ),
                    enhance_prompt=enhance_prompt,
                    tiling_config=tiling_config,
                    max_batch_size=max_batch_size,
                )
                if stage_1_sigmas is not None:
                    pipeline_kwargs["stage_1_sigmas"] = torch.tensor(stage_1_sigmas, dtype=torch.float32)
                if stage_2_sigmas is not None:
                    pipeline_kwargs["stage_2_sigmas"] = torch.tensor(stage_2_sigmas, dtype=torch.float32)
                video, audio = self.pipeline(**pipeline_kwargs)

                encode_video(
                    video=video,
                    fps=frame_rate,
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
