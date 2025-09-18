"""
编码器工厂：根据配置自动选择RGB或RGB+Depth编码器
"""
from typing import Iterable, Optional, Dict, Any
import flax.linen as nn

from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.vision.multimodal_encoding import MultiModalEncodingWrapper
from serl_launcher.vision.segnn_encoder import create_segnn_encoder


def create_encoder_from_config(
    encoder_type: str = "resnet-pretrained",
    use_proprio: bool = False,
    image_keys: Iterable[str] = ("image",),
    # 深度相关配置
    use_depth: bool = False,
    depth_encoder_kwargs: Optional[Dict[str, Any]] = None,
    camera_params: Optional[Dict[str, Dict[str, float]]] = None,
    **kwargs
) -> nn.Module:
    """
    根据配置创建合适的编码器

    Args:
        encoder_type: RGB编码器类型 ("resnet", "resnet-pretrained")
        use_proprio: 是否使用本体感觉
        image_keys: 图像键名列表
        use_depth: 是否使用深度信息
        depth_encoder_kwargs: 深度编码器参数
        camera_params: 相机内参 {camera_name: {fx, fy, cx, cy}}
        **kwargs: 其他参数

    Returns:
        编码器模块 (EncodingWrapper 或 MultiModalEncodingWrapper)
    """

    # 1. 创建RGB编码器
    if encoder_type == "resnet":
        from serl_launcher.vision.resnet_v1 import resnetv1_configs

        rgb_encoders = {
            image_key: resnetv1_configs["resnetv1-10"](
                pooling_method="spatial_learned_embeddings",
                num_spatial_blocks=8,
                bottleneck_dim=256,
                name=f"encoder_{image_key}",
            )
            for image_key in image_keys
        }
    elif encoder_type == "resnet-pretrained":
        from serl_launcher.vision.resnet_v1 import (
            PreTrainedResNetEncoder,
            resnetv1_configs,
        )

        pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
            pre_pooling=True,
            name="pretrained_encoder",
        )
        rgb_encoders = {
            image_key: PreTrainedResNetEncoder(
                pooling_method="spatial_learned_embeddings",
                num_spatial_blocks=8,
                bottleneck_dim=256,
                pretrained_encoder=pretrained_encoder,
                name=f"encoder_{image_key}",
            )
            for image_key in image_keys
        }
    else:
        raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

    # 2. 如果不使用深度，返回标准编码器
    if not use_depth:
        return EncodingWrapper(
            encoder=rgb_encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

    # 3. 使用深度时，创建多模态编码器
    depth_encoder_kwargs = depth_encoder_kwargs or {}
    depth_encoders = {}

    for image_key in image_keys:
        # 获取该相机的内参
        cam_params = camera_params.get(image_key, {}) if camera_params else {}

        depth_encoders[image_key] = create_segnn_encoder(
            camera_params=cam_params if cam_params else None,
            **depth_encoder_kwargs
        )

    return MultiModalEncodingWrapper(
        rgb_encoder=rgb_encoders,
        depth_encoder=depth_encoders,
        use_proprio=use_proprio,
        enable_stacking=True,
        camera_keys=image_keys,
    )


def detect_depth_in_observations(observations: Dict, image_keys: Iterable[str]) -> bool:
    """
    检测观测数据中是否包含深度信息

    支持两种格式：
    1. 独立深度键：depth_camera_name
    2. RGBD拼接：camera_name为4通道数据

    Returns:
        True if depth data is detected
    """
    for image_key in image_keys:
        if image_key in observations:
            # 检查是否存在对应的独立深度键
            depth_key = f"depth_{image_key}"
            if depth_key in observations:
                print(f"检测到独立深度数据: {depth_key}")
                return True

            # 检查图像是否是4通道 (RGBD)
            image = observations[image_key]
            if len(image.shape) >= 3 and image.shape[-1] == 4:
                print(f"检测到4通道RGBD数据: {image_key} {image.shape}")
                return True
    return False


def extract_depth_keys_from_observations(observations: Dict, image_keys: Iterable[str]) -> list[str]:
    """
    从观测数据中提取深度键名

    Returns:
        List of depth keys found in observations
    """
    depth_keys = []
    for image_key in image_keys:
        depth_key = f"depth_{image_key}"
        if depth_key in observations:
            depth_keys.append(depth_key)
    return depth_keys