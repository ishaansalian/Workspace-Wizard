# Workspace Wizard

**Team Members:**  
- Ishaan Salian  
- Aryaman Ghura 
- Kavya Manchanda  
- Mary Esenther

**Project Overview**  
Workspace Wizard is an autonomous desk organization system designed to optimize workspace management by identifying, sorting, and organizing objects on a desk. The system uses a combination of hardware and software to detect object positions, plan efficient placement paths, and execute movements to organize objects with high precision.

**Key Features:**  
- **Object Detection & Recognition:** The system uses a camera and machine learning models to identify and track objects on the desk.
- **Pathfinding Algorithm:** The robot calculates optimal paths for organizing objects, avoiding obstacles and ensuring an efficient layout.
- **Autonomous Movement:** The robot uses motors and tracks to move objects into predefined positions while respecting boundaries and object types.
- **Task Automation:** The robot performs "knolling" (arranging objects neatly and systematically) by aligning objects with a target orientation and spacing.

**Technologies Used:**  
- **Hardware:**  
  - Raspberry Pi / Jetson (for processing and computation)  
  - Stepper motors and tracks (for movement)  
  - Camera (for object detection and tracking)  
  - Power management system (for optimal battery usage)  
- **Software:**  
  - Python (for control logic and algorithms)  
  - OpenCV (for computer vision tasks)  
  - TensorFlow (for machine learning model development)  
  - ROS (Robot Operating System, for communication and task management)  

**Objectives:**  
- Develop and integrate the hardware for object manipulation, including motors and sensor systems.
- Implement pathfinding algorithms and object sorting logic.
- Train and deploy machine learning models to recognize and classify objects.
- Ensure smooth operation with minimal human intervention.

**Current Status:**  
- The hardware setup is largely complete, with the camera, tracks, and motors delivered.
- Progress is being made on integrating the object recognition models and fine-tuning movement algorithms.
- The team is working on refining the pathfinding logic and implementing object placement strategies.

**Upcoming Tasks:**  
- Finalize the software stack and integrate camera input for real-time object detection.
- Begin testing with real objects and ensure the robot can autonomously sort and organize the workspace.

**References:**  
- Hu, Y., Zhang, Z., Zhu, X., Liu, R., Philippe Martin Wyder, & Lipson, H. (2023). Knolling bot: Learning robotic object arrangement from tidy demonstrations. https://api.semanticscholar.org/CorpusID:268513198
