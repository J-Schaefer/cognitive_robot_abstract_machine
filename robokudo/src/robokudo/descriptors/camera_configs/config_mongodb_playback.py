from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Configuration class for MongoDB-based camera data playback.

    This class defines the configuration parameters for reading camera data from a
    MongoDB database, typically used for offline processing or testing with stored
    scene data.

    :ivar interface_type: Type of camera interface, set to "StorageReader"
    :ivar loop: Flag to enable looping over database entries
    :ivar db_name: Name of the MongoDB database to read from
    :ivar semantic_map: Filename of the semantic map configuration

    .. note::
        This configuration is used to read previously stored camera data from a
        MongoDB database, allowing for replay and analysis of recorded scenes.
    """

    # camera
    interface_type: str = "StorageReader"

    # mongo
    loop: bool = True
    db_name: str = "rk_scenes"  # database name
    # tf
    semantic_map: str = "semantic_map_iai_kitchen.yaml"
