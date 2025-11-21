
import pydantic
# INFO: Use strict versions for Int and Float to avoid pydantic coersion.
# To make it so strings dont pass as ints or floats
from pydantic import (
    ConfigDict,
    StrictFloat,
    StrictInt
)

from PIL import Image as pil_image

# Data Display Settings
class ImageDataDisplaySetting(pydantic.BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    rescale_factor: StrictFloat = 1.0
    resampling_method: StrictInt = pil_image.Resampling.NEAREST  # pil_image.Resampling

class TextDataDisplaySetting(pydantic.BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    font_size: StrictInt = 18
    line_height: StrictFloat = 1.25

class AudioDataDisplaySetting(pydantic.BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    loop: bool = False
    playback_rate: StrictFloat = 1.0

class DataDisplaySetting(pydantic.BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    image: ImageDataDisplaySetting = pydantic.Field(default_factory=ImageDataDisplaySetting)
    text: TextDataDisplaySetting = pydantic.Field(default_factory=TextDataDisplaySetting)
    audio: AudioDataDisplaySetting = pydantic.Field(default_factory=AudioDataDisplaySetting)
