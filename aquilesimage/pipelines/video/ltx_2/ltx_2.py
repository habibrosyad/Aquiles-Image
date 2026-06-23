from aquilesimage.utils.utils_video import get_path_file_video_model, file_exists, download_ltx_2, download_ltx_2_3
from typing import Literal
try:
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.types import ModalitySpec
    from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser, SimpleDenoiser
    from ltx_pipelines.utils.helpers import combined_image_conditionings, assert_resolution
    from ltx_pipelines.utils.samplers import euler_denoising_loop, gradient_estimating_euler_denoising_loop
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
except ImportError as e:
    print("Error importing components for LTX-2")
    pass
from PIL import Image
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

    def generate(self, seed: int, prompt: str, save_result_path: str, negative_prompt: str, image=None, seconds=None, height=1088, width=1920, num_inference_steps=25, use_gradient_estimation=True, ge_gamma=2.0):
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
                dtype = torch.bfloat16

                images_list = self._normalize_images(image, num_frames)
                p = self.pipeline

                gen = torch.Generator(device=p.device).manual_seed(seed)
                noiser = GaussianNoiser(generator=gen)

                ctx_p, ctx_n = p.prompt_encoder(
                    [prompt, negative_prompt],
                    enhance_first_prompt=False,
                    enhance_prompt_image=images_list[0][0] if len(images_list) > 0 else None,
                    enhance_prompt_seed=seed,
                )
                v_ctx_p, a_ctx_p = ctx_p.video_encoding, ctx_p.audio_encoding
                v_ctx_n, a_ctx_n = ctx_n.video_encoding, ctx_n.audio_encoding

                h2, w2 = self._align_resolution(height // 2, width // 2, 16)
                stage_1_half = type("s", (), {"height": h2, "width": w2})()
                stage_1_cond = p.image_conditioner(
                    lambda enc: combined_image_conditionings(
                        images=images_list,
                        height=stage_1_half.height,
                        width=stage_1_half.width,
                        video_encoder=enc,
                        dtype=dtype,
                        device=p.device,
                    )
                )

                sigmas = p._scheduler.execute(steps=num_inference_steps).to(dtype=torch.float32, device=p.device)
                loop_fn = gradient_estimating_euler_denoising_loop if use_gradient_estimation else euler_denoising_loop

                video_state, audio_state = p.stage_1(
                    denoiser=FactoryGuidedDenoiser(
                        v_context=v_ctx_p,
                        a_context=a_ctx_p,
                        video_guider_factory=create_multimodal_guider_factory(
                            params=MultiModalGuiderParams(cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7, modality_scale=1.0, skip_step=0, stg_blocks=[29]),
                            negative_context=v_ctx_n,
                        ),
                        audio_guider_factory=create_multimodal_guider_factory(
                            params=MultiModalGuiderParams(cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7, modality_scale=1.0, skip_step=0, stg_blocks=[29]),
                            negative_context=a_ctx_n,
                        ),
                    ),
                    sigmas=sigmas,
                    noiser=noiser,
                    width=stage_1_half.width,
                    height=stage_1_half.height,
                    frames=num_frames,
                    fps=self.FRAME_RATE,
                    video=ModalitySpec(context=v_ctx_p, conditionings=stage_1_cond),
                    audio=ModalitySpec(context=a_ctx_p),
                    loop=loop_fn,
                    stepper=EulerDiffusionStep(),
                    max_batch_size=1,
                )

                upscaled_video = p.upsampler(video_state.latent[:1])

                stage_2_cond = p.image_conditioner(
                    lambda enc: combined_image_conditionings(
                        images=images_list,
                        height=height,
                        width=width,
                        video_encoder=enc,
                        dtype=dtype,
                        device=p.device,
                    )
                )

                from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMAS
                stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=p.device)

                video_state, audio_state = p.stage_2(
                    denoiser=SimpleDenoiser(v_context=v_ctx_p, a_context=a_ctx_p),
                    sigmas=stage_2_sigmas,
                    noiser=noiser,
                    width=width,
                    height=height,
                    frames=num_frames,
                    fps=self.FRAME_RATE,
                    video=ModalitySpec(
                        context=v_ctx_p,
                        conditionings=stage_2_cond,
                        noise_scale=stage_2_sigmas[0].item(),
                        initial_latent=upscaled_video,
                    ),
                    audio=ModalitySpec(
                        context=a_ctx_p,
                        noise_scale=stage_2_sigmas[0].item(),
                        initial_latent=audio_state.latent,
                    ),
                    loop=euler_denoising_loop,
                    stepper=EulerDiffusionStep(),
                    max_batch_size=1,
                )

                decoded_video = p.video_decoder(video_state.latent, tiling_config, gen)
                decoded_audio = p.audio_decoder(audio_state.latent)

                encode_video(
                    video=decoded_video,
                    fps=self.FRAME_RATE,
                    audio=decoded_audio,
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