import os
import sys
import yaml
import argparse
import logging
from omegaconf import OmegaConf 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import VideoGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

def load_configurations():
    """
    3 farklı yaml dosyasını okur ve tek bir sözlükte (dictionary) birleştirir.
    """
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    
    paths_cfg_path = os.path.join(config_dir, "paths.yaml")
    runtime_cfg_path = os.path.join(config_dir, "runtime.yaml")
    sdxl_cfg_path = os.path.join(config_dir, "sdxl_video.yaml")

    with open(paths_cfg_path, 'r') as f:
        paths_conf = yaml.safe_load(f)
        
    with open(runtime_cfg_path, 'r') as f:
        runtime_conf = yaml.safe_load(f)
        
    with open(sdxl_cfg_path, 'r') as f:
        model_conf = yaml.safe_load(f)

    base_dir = paths_conf['paths']['base_dir']
    for key, val in paths_conf['paths']['models'].items():
        if isinstance(val, str):
            paths_conf['paths']['models'][key] = val.replace("${paths.base_dir}", base_dir)
            
    for key, val in paths_conf['paths']['output'].items():
         if isinstance(val, str):
            paths_conf['paths']['output'][key] = val.replace("${paths.base_dir}", base_dir)

    full_config = {}
    full_config.update(paths_conf)
    full_config.update(runtime_conf)
    full_config.update(model_conf)

    return full_config

def main():
    logger.info("Konfigürasyon dosyaları okunuyor...")
    config = load_configurations()
    
    generator = VideoGenerator(config)
    
    user_prompt = "A cinematic drone shot of a futuristic city at sunset, cyberpunk style, 8k"
    
    frames = generator.generate(
        prompt=user_prompt,
        negative_prompt=config['model']['generation']['negative_prompt'],
        num_frames=config['model']['generation']['num_frames'],
        width=config['model']['generation']['width'],
        height=config['model']['generation']['height'],
        seed=config['runtime']['seed']
    )
    
    output_path = os.path.join(config['paths']['output']['videos'], "test_video.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    from utils.video_utils import save_video_frames

    if frames:
        save_video_frames(frames, output_path, fps=config['model']['generation']['fps'])
        logger.info(f"Video başarıyla kaydedildi: {output_path}")
    else:
        logger.warning("Üretilen frame olmadığı için video kaydedilmedi.")

if __name__ == "__main__":
    main()