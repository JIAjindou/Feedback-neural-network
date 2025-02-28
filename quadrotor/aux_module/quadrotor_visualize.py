import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

class Quadrotor_Visualize:
    def __init__(self):
        self.r = 0.07
        self.l = 0.25
        self.propeller_pos = np.array(
            [
                [np.sqrt(2) / 4 * self.l, np.sqrt(2) / 4 * self.l, 0],
                [-np.sqrt(2) / 4 * self.l, np.sqrt(2) / 4 * self.l, 0],
                [-np.sqrt(2) / 4 * self.l, -np.sqrt(2) / 4 * self.l, 0],
                [np.sqrt(2) / 4 * self.l, -np.sqrt(2) / 4 * self.l, 0],
            ]
        )  # 4 propeller center pos

        # Plot Param
        self.prop_alpha = 0.6
        self.bodyX_width = 1.0
        self.bodyX_alpha = 0.5
        self.bodyX_color = "k"
        self.bodyX_arralpha = 1.0
        self.bodyX_arrlen = 0.2
        self.bodyX_arrwidth = 0.2

    def euler_to_Rbe(self, roll, pitch, yaw):
        # Input with Rad
        # return earth to body point mapping
        R_psi = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )
        R_theta = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )
        R_phi = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )
        dcm_eb = R_psi @ R_theta @ R_phi
        return dcm_eb.T

    def quat_to_DCMeb(self, Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                This rotation matrix converts a point in the local reference
                frame to a point in the global reference frame.

        DCMeb: rotate earth frame to body frame == converts a point in body frame 
                                                    to a point in the earth frame.
        """
        # Quaternion Normalization
        Q = Q / np.linalg.norm(Q)

        # Extract the values from Q
        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 3x3 Matrix [xb, yb, zb]
        # [[r00, r01, r02],
        # [r10, r11, r12],
        # [r20, r21, r22]]
        Reb = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

        return Reb
    
    def plot_propeller(self, ax, pos, Rbe, pc_list):
        a = np.linspace(0, 2 * np.pi)
        x = self.r * np.cos(a)
        y = self.r * np.sin(a)
        z = np.zeros_like(a)
        xyz = np.vstack([x, y, z]).T
        xyz_rotate = np.array([Rbe @ xyzi for xyzi in xyz])
        pos_rotate = np.array([Rbe @ prop_pos for prop_pos in self.propeller_pos])
        for i, singleprop_pos in enumerate(pos_rotate):
            verts = [
                list(
                    zip(
                        xyz_rotate[:, 0] + pos[0] + singleprop_pos[0],
                        xyz_rotate[:, 1] + pos[1] + singleprop_pos[1],
                        xyz_rotate[:, 2] + pos[2] + singleprop_pos[2],
                    )
                )
            ]
            ax.add_collection3d(
                Poly3DCollection(verts, facecolors=pc_list[i], alpha=self.prop_alpha)
            )

    def plot_bodyX(self, ax, pos, Rbe, c):
        # Plot X
        x1 = np.linspace(-np.sqrt(2) / 4 * self.l, np.sqrt(2) / 4 * self.l)
        y1 = np.linspace(-np.sqrt(2) / 4 * self.l, np.sqrt(2) / 4 * self.l)
        z1 = np.zeros_like(x1)
        xyz1 = np.vstack([x1, y1, z1]).T
        xyz_rotate1 = np.array([Rbe @ xyzi for xyzi in xyz1])
        ax.plot3D(
            xyz_rotate1[:, 0] + pos[0],
            xyz_rotate1[:, 1] + pos[1],
            xyz_rotate1[:, 2] + pos[2],
            c=c,
            alpha=self.bodyX_alpha,
            linewidth=self.bodyX_width,
        )

        x2 = np.linspace(-np.sqrt(2) / 4 * self.l, np.sqrt(2) / 4 * self.l)
        y2 = np.linspace(np.sqrt(2) / 4 * self.l, -np.sqrt(2) / 4 * self.l)
        z2 = np.zeros_like(x2)
        xyz2 = np.vstack([x2, y2, z2]).T
        xyz_rotate2 = np.array([Rbe @ xyzi for xyzi in xyz2])
        ax.plot3D(
            xyz_rotate2[:, 0] + pos[0],
            xyz_rotate2[:, 1] + pos[1],
            xyz_rotate2[:, 2] + pos[2],
            c=c,
            alpha=self.bodyX_alpha,
            linewidth=self.bodyX_width,
        )

        # Plot Arrow
        direction = Rbe @ np.array([0, 0, -np.sqrt(2) / 8 * self.l])
        direction = direction / np.linalg.norm(direction) * self.bodyX_arrlen
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            direction[0],
            direction[1],
            direction[2],
            color=self.bodyX_color,
            alpha=self.bodyX_arralpha,
            linewidth=self.bodyX_arrwidth,
        )

    def plot_quadrotorEul(self, ax, x, pc_list, bc):
        pos = x[0:3]
        euler_ang = -x[6:9]
        # Visualize a quadrotor with attitude
        self.plot_propeller(ax, pos, self.euler_to_Rbe(*euler_ang), pc_list=pc_list)
        self.plot_bodyX(ax, pos, self.euler_to_Rbe(*euler_ang), c=bc)

    def plot_quadrotorQuat(self, ax, x, pc_list, bc):
        pos = x[0:3]
        quat = x[6:10]
        quat = np.array([quat[0],-quat[1],-quat[2],-quat[3]])
        # Visualize a quadrotor with attitude
        self.plot_propeller(ax, pos, self.quat_to_DCMeb(quat).T, pc_list=pc_list)
        self.plot_bodyX(ax, pos, self.quat_to_DCMeb(quat).T, c=bc)

# if __name__ == "__main__":
#     vis = Quadrotor_Visualize()
#     fig1 = plt.figure()
#     ax = fig1.add_subplot(projection="3d")
#     ax.set_aspect("equal")
#     vis.plot_quadrotorEul(ax, x=np.zeros(12), pc_list=["k"] * 4, bc="k")
#     plt.show()
