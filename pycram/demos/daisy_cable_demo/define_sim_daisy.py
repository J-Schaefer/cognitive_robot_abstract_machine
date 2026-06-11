from pycram.datastructures.dataclasses import Context
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.daisy import DAiSy


def setup_sim_daisy():
    # %% Robot Setup
    daisy = "package://iai_daisy_description/robots/daisy.urdf.xacro"
    daisy_parser = URDFParser.from_file(file_path=daisy)
    daisy_world = daisy_parser.parse()
    DAiSy.from_world(daisy_world)

    world = daisy_world

    # Robot semantic view
    robot_view = world.get_semantic_annotations_by_type(DAiSy)[0]

    # Context
    context = Context(
        world,
        robot_view,
    )

    return world, robot_view, context