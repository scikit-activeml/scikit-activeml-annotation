
import pydantic
from PIL import Image as pil_image

# Data Display Settings
class ImageDataDisplaySetting(pydantic.BaseModel):
    rescale_factor: float = 1
    resampling_method: pil_image.Resampling = pil_image.Resampling.NEAREST
    
class TextDataDisplaySetting(pydantic.BaseModel):
    font_size: int = 18
    line_height: float = 1.25

class AudioDataDisplaySetting(pydantic.BaseModel):
    pass

class DataDisplaySetting(pydantic.BaseModel):
    image: ImageDataDisplaySetting = pydantic.Field(default_factory=ImageDataDisplaySetting)
    text: TextDataDisplaySetting = pydantic.Field(default_factory=TextDataDisplaySetting)
    audio: AudioDataDisplaySetting = pydantic.Field(default_factory=AudioDataDisplaySetting)
