# Midterm: Superpixel Segmentation for Climbing Walls
## Prof. Jacky Baltes <jacky.baltes@ntnu.edu.tw>
## Prof. Reinhard Gerndt <r.gerndt@ostfalia.de>
## Due Date: 2nd November 2026 - 23:59 Taiwan Time

# Midterm

This project is part of the midterm examination in the course Robot and Computer Vision at the National Taiwan Normal University (NTNU) in Taiwan.

This project must be done individually.

After the submission of your project, you will demonstrate your knowledge of the material in an oral examination. 
The oral exam will include questions about this project as well as any other material covered in class.

# Introduction

The goal of this midterm is to evaluate the performance of several superpixel algorithms in the domain of rock climbing for humanoid robots. See [rock-climbing-robots.github.io](https://rock-climbing-robots.github.io) for more information.

In particular, we investigate the histogram based algorithm of Van der BErgh et al [SEEDS: Superpixels Extracted via Energy Driven Sample](https://arxiv.org/abs/1309.3848).

The initial step in climbing a route is to extract the set of legal handholds of the wall. This step can be done using either a global vision or the robot's camera when the robot is standing far enough away from the wall.

In this assignment, the goal is to extract the type, position, and orientation of various colored holds in an image. The orientation of the hold is the angle of the holding edge (i.e., longest straight edge of the hold). The evaluation will include both images from real climbing walls as well as synthetic images created as part of the [perception challenge](https://jackybaltes.github.io/rcr-perception/ThreeJS/) of the 1st rock climbing robot competition.

|  |  |
| --- | --- |
| ![Climbing wall image 1](https://i.postimg.cc/SKbTjRkj/image.png) | ![Climbing wall image 2](https://i.postimg.cc/BQ1BD3h5/image.png) |
| ![Climbing wall image 3](https://i.postimg.cc/c4Y5WfGb/image.png) | ![Climbing wall image 4](https://i.postimg.cc/6ptzNS08/image.png) |

# Superpixels

The varied shapes and colors of the climbing holds pose a particular challenge for object detection. Superpixels (i.e., regions defined by local texture and boundary) seem to be specifically suited to this problem.

In this project, you must compare the performance of your the various algorithms and your approach to parameter settings for the following superpixel algorithms

    1.) SuperpixelLSC
    2.) SuperpixelSEEDS
    3.) SuperpixelSLIC

These algorithms are implemented as part of the [OpenCV ximgproc module](https://docs.opencv.org/4.0.0/df/d6c/group__ximgproc__superpixel.html)

# Output
Your algorithm must return

    type: the type/class of hold. In the synthetic images, all holds have a a specified type, in the real images, you want to look for similar holds in the image and assign them a label.

    position: the x,y,z coordinates of the center of the hold. You can calibrate the view with the red square in the synthetic images. The bottom left corner of the square is at the origin and the size of the red square is 1m. The spacing of the wall grid is 20cm in the real images.

    orientation: the pitch, roll, and yaw of the hold with respect to the world coordinate system.  

# Evaluation

You are expected to carefully evaluate the performance of your vision system. 

Firstly, desrcibe your approach for selecting the best parameter settings for the various algorithms.  
Secondly, show the sensitivity of your system with respect to the parameters. Plot a curve showing the performance of your algorithm with at least one crucial parameter (ROC curve).

For the oral exam, you should prepare convincing evidence of your system running and its performance.

# Report

Create a 2 to 5 page report describing your approach to solving this project. Explain why you chose this particular approach (e.g., CNN, color detection), how you structured your code, what major issues did you face during implementation, and how you evaluated the system.

# Marking

Due to the increased capabilities of current state of art AI systems and the increased use of these systems by students, the submission of the project will be marked as pass/fail woth 1% of the final grade.

You must demonstrate the capabilties of your system and answer questions about your submission in an oral exam after the submission deadline, which is worth 19% of the final grade.
