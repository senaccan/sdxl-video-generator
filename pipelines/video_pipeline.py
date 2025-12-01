import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel, SchedulerMixin
from typing import List, Optional, Union
from PIL import Image

class SDXLVideoPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "unet->vae"

    def __init__(self, vae: AutoencoderKL, unet: UNet2DConditionModel, scheduler: SchedulerMixin):
        super().__init__()
        
        self.register_modules(vae=vae, unet=unet, scheduler=scheduler)

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 16,
        width: int = 1024,
        height: int = 576,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ):
        device = self._execution_device
        
        print(f"Pipeline çalışıyor... {num_inference_steps} adım işlenecek.")
        
        latents = torch.randn(
            (num_frames, 4, height // 8, width // 8),
            generator=generator,
            device=device,
            dtype=self.unet.dtype
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        print("VAE Decoding işlemi yapılıyor...")

        latents = latents / self.vae.config.scaling_factor
        
        image_list = []
        with torch.no_grad():
            for i in range(latents.shape[0]):
                single_latent = latents[i : i + 1]
                
                image = self.vae.decode(single_latent).sample

                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                image_list.append(Image.fromarray((image[0] * 255).astype("uint8")))

        return type("Output", (object,), {"frames": image_list})