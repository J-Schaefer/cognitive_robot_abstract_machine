# Motion Statecharts

Motion Statecharts are a core concept in Giskard for composing complex robot motions. They provide a structured way to manage the transition between different motion goals and monitors, making it easier to build robust and reactive robot behaviors.

## The Problem

Traditional robot motion planning often involves a sequence of fixed waypoints or a single, monolithic trajectory. This approach faces several challenges:

- **Complex Sequencing**: Coordinating multiple movements (e.g., "move arm to pre-grasp," then "close gripper," then "lift arm") can become hard to manage as the number of steps increases.
- **Error Handling**: What happens if a collision is detected mid-motion? Or if the gripper fails to close? Handling these contingencies in a flat script often leads to "spaghetti code."
- **Reactivity**: Modern robots need to respond to their environment. A simple trajectory doesn't easily allow for behavior like "move until a certain force is felt" or "stop if a human enters the workspace."

## How Motion Statecharts Solve It

Motion Statecharts address these issues by using a state machine-based approach to motion composition. 

### Key Concepts

- **Nodes**: Every state in the statechart is a node. Nodes can be **Goals** (which define what the robot should do) or **Monitors** (which check for specific conditions).
- **Transitions**: Transitions define the flow of execution. A transition is triggered by a condition (usually from a Monitor) and leads to another Node or ends the motion.
- **Hierarchical Composition**: Statecharts can contain other statecharts, allowing for complex behaviors to be built from simpler components.
- **Parallel and Sequential Execution**: Using nodes like `Parallel` and `Sequence`, you can easily specify which goals should be pursued at the same time and which must follow one another.

### Benefits

- **Modularity**: Individual motions and checks are self-contained nodes that can be reused across different tasks.
- **Clarity**: The statechart structure provides a clear visual and logical representation of the robot's behavior.
- **Robustness**: Error handling and environment reactivity are built directly into the motion's structure through monitors and transitions.
- **Constraint-Based**: Because Giskard is constraint-based, multiple goals in a `Parallel` node are solved together, ensuring the robot satisfies all requirements simultaneously (e.g., "reach for the cup while keeping the arm away from the table").

For practical examples of how to use Motion Statecharts, see the [Basic Motion](examples/basic_motion.md) and [Cartesian Goals](examples/cartesian_goals.md) tutorials.
