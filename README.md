# SE(3) Particle Filter for Monocular Camera CNN Pose Estimates

This is the readme file for a term project titled "SE(3) Particle Filter for Monocular Camera CNN Pose Estimates". The purpose of this project is to explore the application of a particle filter to improve the accuracy of CNN pose estimates for a monocular camera.

## Running the code:
To run our code, first ensure you have the following Python libraries installed (i.e. ```pip install <libary name>```:
- ```numpy```
- ```scipy```
- ```transforms3D```
- ```matplotlib```

After installing the above Python libraries, simply download the contents of the repository and run ```main.py```.

## Introduction
Pose estimation is an important task in computer vision and robotics. It involves determining the position and orientation of an object in a 3D space. One common method for pose estimation is using a monocular camera and a convolutional neural network (CNN). The one we use in this project is [produced by Dr. Maani Ghaffari's research lab](https://ieeexplore-ieee-org.proxy.lib.umich.edu/document/9672748) [1]. However, CNNs can have difficulty accurately estimating poses in certain situations, such as when objects are occluded or the camera is moving rapidly. To address these issues, we propose the use of a particle filter to improve the accuracy of the CNN pose estimates.

## CNN
The CNN used in this project is a standard architecture for pose estimation. It takes an input image and produces a homogeneous transformation matrix representing the position and orientation of the object in 3D space. We train the CNN on a dataset of synthetic images with known ground truth poses.

## Particle Filter
The particle filter is a Bayesian filtering technique that is commonly used for state estimation in robotics. In this project, we use an SE(3) particle filter to estimate the pose of the object. The particle filter uses a set of particles to represent the distribution of the object's pose. The particles are updated based on the CNN pose estimate, which receives images from the monocular camera.

__Particle Filter Diagram:__

![alt text](/pictures/pf_diagram.PNG)

## Results
We evaluate the performance of the SE(3) particle filter on a dataset of real-world images. We compare the accuracy of the particle filter to the accuracy of the CNN alone. Our results show that the particle filter significantly improves the accuracy of the pose estimates, particularly in situations where the CNN struggles.

__Errors:__

![alt text](/pictures/Figure_2.png)
![alt text](/pictures/Figure_3.png)

__3D Visualization:__

![alt text](/pictures/Figure_1.png)
![alt text](/pictures/Figure_1_Zoom.png)


## Conclusions
In this project, we have demonstrated the potential of using a particle filter to improve the accuracy of CNN pose estimates for a monocular camera. Our results suggest that the particle filter can be a useful tool for pose estimation in real-world applications. Further research is needed to explore the performance of the particle filter in different settings and with different CNN architectures.

## References

[1] J. Song, M. Patel, M. Ghaffari. “Fusing Convolutional Neural Network and Geometric Constraint for Image-Based Indoor Localization” in *IEEE Robotics and Automation Letters*, vol. 7, no. 2, pp. 1674-1681, April 2022, doi: 10.1109/LRA.2022.3140832.
