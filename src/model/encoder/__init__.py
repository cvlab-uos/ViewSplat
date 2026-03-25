from typing import Optional

from .encoder import Encoder
from .visualization.encoder_visualizer import EncoderVisualizer
from .encoder_spf_viewsplat import EncoderSPF_ViewSplatCfg, EncoderSPF_ViewSplat
from .encoder_spfv2_viewsplat import EncoderSPFV2_ViewSplatCfg, EncoderSPFV2_ViewSplat
from .encoder_spfv2l_viewsplat import EncoderSPFV2L_ViewSplatCfg, EncoderSPFV2L_ViewSplat

ENCODERS = {
    "spf_viewsplat": (EncoderSPF_ViewSplat, None),
    "spfv2_viewsplat": (EncoderSPFV2_ViewSplat, None),
    "spfv2l_viewsplat": (EncoderSPFV2L_ViewSplat, None),
}

EncoderCfg = EncoderSPF_ViewSplatCfg | EncoderSPFV2_ViewSplatCfg | EncoderSPFV2L_ViewSplatCfg 

def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
