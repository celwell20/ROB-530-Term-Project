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
Pose estimation is an important task in computer vision and robotics. It involves determining the position and orientation of an object in a 3D space. One common method for pose estimation involves using a monocular camera and a convolutional neural network (CNN). The one selected for this project is [produced by Dr. Maani Ghaffari's research lab](https://ieeexplore-ieee-org.proxy.lib.umich.edu/document/9672748) [1]. To further improve this pose estimate, we propose introducing a particle filter which will propagate a uniform distribution of SE(3) particles through the same motion model used by the CNN, whilst updating their weights based on the CNN-based estimation.

## CNN
The CNN used as a starting point for this project presents a standard architecture for pose estimation. It takes an input image and produces a homogeneous transformation matrix representing the position and orientation of the object in 3D space. For this project, we assume these pose estimates are stored, alongside the ground truth data, in .npy lists of 4x4 arrays saved in the "data" subfolder.

## Particle Filter
Particle filters are a Bayesian filtering technique that is commonly used for state estimation in robotics. In this project, the particle filter uses a set of SE(3) poses as particles to represent the distribution of the object's pose. The motion model is the same constant-velocity motion model employed in [1], whilst the measurment model is represented by the CNN-based localization algoirthm.

__Particle Filter Diagram:__

![alt text](/pictures/pf_diagram.PNG)

## Results
We evaluate the performance of the SE(3) particle filter on a single dataset of real-world images. We compare the accuracy of the particle filter to the accuracy of the CNN-based estimates.

__Errors:__

![alt text](/pictures/Figure_2.png)
![alt text](/pictures/Figure_3.png)

__3D Visualization:__

![alt text](/pictures/Figure_1.png)
![alt text](/pictures/Figure_1_Zoom.png)


## Conclusions
In this project, we have demonstrated the potential of using a particle filter to improve the accuracy of CNN-based pose estimates for a monocular camera. Our results suggest that the particle filter is particularly good at estimating the translational components of the pose, whilst its rotational counterparts are not always as accurate. Further research is needed to explore the performance of the particle filter in different settings and with different CNN architectures.

## References

[1] J. Song, M. Patel, M. Ghaffari. “Fusing Convolutional Neural Network and Geometric Constraint for Image-Based Indoor Localization” in *IEEE Robotics and Automation Letters*, vol. 7, no. 2, pp. 1674-1681, April 2022, doi: 10.1109/LRA.2022.3140832.

## Authors
Clayton Elwell, Yi Shen, Jiangbo Yu, Federico Seghizzi, Lucas Lymburner