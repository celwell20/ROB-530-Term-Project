import numpy as np
from scipy.spatial.transform import Rotation

class SE3:
    def __init__(self, position=np.zeros(3), rotation=np.eye(3)):
        # Construct an SE(3) object based on a provided rotation matrix and position vector.
        self.position = position
        self.rotation = rotation

    def pose(self):
        # Return the pose
        pose = np.row_stack((np.column_stack((self.rotation, self.position.T)),[0, 0, 0, 1]))
        return pose


def read(fileName):
    measurements = []
    with open(fileName,'r') as file:
        contents = file.readlines()
        for i in range(1662,1761):
            # position = np.array(contents[i][])
            line = contents[i].split()
            position = np.array([float(line[3]),float(line[4]),float(line[5])])
            rotation = np.array([float(line[6]),float(line[7]),float(line[8]),float(line[9])])
            rotation = Rotation.from_quat(rotation).as_matrix()
            covariance = np.eye(6)
            covariance[0,:] = np.array([float(line[10]),float(line[11]),float(line[12]),float(line[13]),float(line[14]),float(line[15])])
            covariance[1,1:6] = np.array([float(line[16]),float(line[17]),float(line[18]),float(line[19]),float(line[20])])
            covariance[2,2:6] = np.array([float(line[21]),float(line[22]),float(line[23]),float(line[24])])
            covariance[3,3:6] = np.array([float(line[25]),float(line[26]),float(line[27])])
            covariance[4,4:6] = np.array([float(line[28]),float(line[29])])
            covariance[5,5:6] = np.array([float(line[30])])
            covariance += covariance.T - np.diag(covariance.diagonal())
            measurements.append(SE3(position, rotation).pose())

        # line = contents[1663].split()
        # print(measurements[1].pose())
        # print(position)
        # print(covariance)
        return measurements, covariance

def readThis(fileName):
    measurements = []
    with open(fileName,'r') as file:
        contents = file.readlines()
        for content in contents:
            line = content.split()
            position = np.array([float(line[3]),float(line[7]),float(line[11])])
            rotation = np.array([[float(line[0]), float(line[1]) , float(line[2])] ,
                              [float(line[4]),float(line[5]),float(line[6])],
                              [float(line[8]),float(line[9]),float(line[10])]])
            measurements.append(SE3(position, rotation).pose())
    return measurements
        # print(contents[0])

def main():
    # read('parking-garage.g2o')
    measurements = readThis('output.txt')
    return

if __name__ == '__main__':
    main()