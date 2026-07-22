import os

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.daisy import DAiSy, DAiSyLeftArm, DAiSyRightArm


def test_daisy_left_and_right_arms_are_distinct():
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "semantic_digital_twin",
        "resources",
        "urdf",
    )
    daisy_path = os.path.join(urdf_dir, "daisy.urdf")
    daisy_world = URDFParser.from_file(file_path=daisy_path).parse()
    DAiSy.from_world(daisy_world)
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]

    assert isinstance(daisy.left_arm, DAiSyLeftArm)
    assert isinstance(daisy.right_arm, DAiSyRightArm)
    assert daisy.left_arm is not daisy.right_arm
