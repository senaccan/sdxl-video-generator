import torch
import logging
import sys
import os
from typing import Optional, Dict, Any, List

from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from diffusers import UNet2DConditionModel as OfficialUNet 
from diffusers.utils import logging as diffusers_logging

from utils.model_utils import get_device, flush_vram
from pipelines.video_pipeline import SDXLVideoPipeline

try:
    from models.unet.unet_base import UNet2DConditionModel as CustomUNet
except ImportError as e:
    print(f"HATA: models/unet/unet_base.py dosyası bulunamadı: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

diffusers_logging.set_verbosity_error()

class VideoGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device(force_cpu=False)
        
        precision = config['runtime'].get('precision', 'fp16')
        self.dtype = torch.float16 if precision == 'fp16' else torch.float32
        
        self.pipeline = None
        self._load_models()

    def _load_models(self):
        logger.info(f"Modeller yükleniyor... Cihaz: {self.device}, Tip: {self.dtype}")
        
        try:
            unet_path = self.config['paths']['models']['sdxl_base']
            vae_path = self.config['paths']['models']['sdxl_vae']

            if not os.path.exists(unet_path):
                raise FileNotFoundError(f"UNet modeli bulunamadı: {unet_path}")
                
            logger.info("1/3: Orijinal SDXL ağırlıkları okunuyor...")

            temp_unet = OfficialUNet.from_single_file(
                unet_path, 
                torch_dtype=self.dtype
            )

            logger.info("2/3: Ağırlıklar Custom UNet (unet_base.py) mimarisine aktarılıyor...")
            
            custom_unet = CustomUNet(**temp_unet.config)
            
            custom_unet.load_state_dict(temp_unet.state_dict())
            
            custom_unet.to(dtype=self.dtype)
            
            del temp_unet
            flush_vram()

            logger.info(f"3/3: VAE yükleniyor: {vae_path}")
            if not os.path.exists(vae_path):
                 raise FileNotFoundError(f"VAE bulunamadı: {vae_path}")

            vae = AutoencoderKL.from_single_file(
                vae_path, 
                torch_dtype=self.dtype
            )

            scheduler_config = self.config['model'].get('scheduler', {})
            scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)

            logger.info("Pipeline kuruluyor...")

            self.pipeline = SDXLVideoPipeline(
                vae=vae,
                unet=custom_unet,
                scheduler=scheduler 
            )
            
            self.pipeline.to(self.device)

            if self.config['runtime'].get('enable_model_cpu_offload', False):
                self.pipeline.enable_model_cpu_offload()

            logger.info(" Tüm modeller başarıyla yüklendi.")

        except Exception as e:
            logger.error(f" Model yükleme hatası: {e}")
            import traceback
            traceback.print_exc()
            self.pipeline = None

    def generate(self, prompt, negative_prompt="", num_frames=24, width=1024, height=576, seed=None):
        if self.pipeline is None:
            logger.error("Pipeline hazır değil. İşlem iptal edildi.")
            return []

        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(f" Üretim Başlıyor: '{prompt}'")

        try:
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                width=width,
                height=height,
                num_inference_steps=self.config['model']['generation']['steps'],
                guidance_scale=self.config['model']['generation']['guidance_scale'],
                generator=generator
            )

            if self.config['runtime'].get('flush_vram_after_generation', True):
                flush_vram()
                
            return output.frames

        except Exception as e:
            logger.error(f"Generate hatası: {e}")
            import traceback
            traceback.print_exc()
            return []