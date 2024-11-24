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

**References:**  
- Hu, Y., Zhang, Z., Zhu, X., Liu, R., Philippe Martin Wyder, & Lipson, H. (2023). Knolling bot: Learning robotic object arrangement from tidy demonstrations. https://api.semanticscholar.org/CorpusID:268513198
