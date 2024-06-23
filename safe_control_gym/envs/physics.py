from __future__ import annotations

import numpy as np
from enum import Enum
import numpy.typing as npt

from safe_control_gym.envs.drone import Drone

import pybullet as p
from typing import NamedTuple
from scipy.spatial.transform import Rotation as R


class Constants:
    GRAVITY: float = 9.81


ForceTorque = NamedTuple("ForceTorque", f=npt.NDArray[np.float_], t=npt.NDArray[np.float_])


class PhysicsMode(str, Enum):
    """Physics implementations enumeration class."""

    PYB = "pyb"  # Base PyBullet physics update.
    DYN = "dyn"  # Update with an explicit model of the dynamics.
    PYB_GND = "pyb_gnd"  # PyBullet physics update with ground effect.
    PYB_DRAG = "pyb_drag"  # PyBullet physics update with drag.
    PYB_DW = "pyb_dw"  # PyBullet physics update with downwash.
    # PyBullet physics update with ground effect, drag, and downwash.
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw"


class Physics:
    def __init__(self, pyb_client: int, dt: float, mode: PhysicsMode = PhysicsMode.PYB):
        self.pyb_client = pyb_client
        self.dt = dt
        self.mode = PhysicsMode(mode)
        assert isinstance(self.mode, PhysicsMode), "mode must be a PhysicsMode."

    def calculate_force_torques(
        self, drone: Drone, rpms: npt.NDArray[np.float_], other_drones: list[Drone]
    ) -> dict[int, list[str, npt.NDArray[np.float_]]]:
        """Physics update function.

        Args:
            drone: The target drone to calculate the physics for.
            rpms: The rpms to apply to the drone rotors.
            other_drones: List of other drones in the environment.
        """
        if self.mode == PhysicsMode.PYB:
            return self.motors(drone, rpms)
        elif self.mode == PhysicsMode.DYN:
            return self.dynamics(drone, rpms)
        elif self.mode == PhysicsMode.PYB_GND:
            return self.motors(drone, rpms) + self.ground_effect(drone, rpms)
        elif self.mode == PhysicsMode.PYB_DRAG:
            return self.motors(drone, rpms) + self.drag(drone, rpms)
        elif self.mode == PhysicsMode.PYB_DW:
            return self.motors(drone, rpms) + self.downwash(drone, other_drones)
        elif self.mode == PhysicsMode.PYB_GND_DRAG_DW:
            return (
                self.motors(drone, rpms)
                + self.drag(drone, rpms)
                + self.downwash(drone, other_drones)
            )
        raise NotImplementedError(f"Physics mode {self.mode} not implemented.")

    def apply(
        self,
        drone: Drone,
        force_torques: list[tuple[int, ForceTorque]],
        external_force: npt.NDArray[np.float_] | None = None,
    ):
        """Apply the calculated forces and torques in simulation.

        Args:
            drone: The target drone to apply the forces and torques to.
            force_torques: A dictionary of forces and torques for each link of the drone body.
            external_force: An optional, external force to apply to the drone body.
        """
        for link_id, ft in force_torques:
            p.applyExternalForce(
                drone.id,
                link_id,
                forceObj=ft.f,
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.pyb_client,
            )
            p.applyExternalTorque(
                drone.id,
                link_id,
                torqueObj=ft.t,
                flags=p.LINK_FRAME,
                physicsClientId=self.pyb_client,
            )
        if external_force is not None:
            p.applyExternalForce(
                drone.id,
                linkIndex=4,  # Link attached to the quadrotor's center of mass.
                forceObj=external_force,
                posObj=drone.state["pos"],
                flags=p.WORLD_FRAME,
                physicsClientId=self.pyb_client,
            )

    def step(self, drone: Drone):
        """Advance the simulation by one step."""
        # PyBullet computes the new state, unless PhysicsMode.DYN.
        if self.mode != PhysicsMode.DYN:
            p.stepSimulation(physicsClientId=self.pyb_client)
        elif self.mode == PhysicsMode.DYN:
            p.resetBasePositionAndOrientation(
                drone.id,
                drone.state["pos"],
                p.getQuaternionFromEuler(drone.state["rpy"]),
                physicsClientId=self.pyb_client,
            )
            p.resetBaseVelocity(
                drone.id,
                drone.state["vel"],
                drone.state["ang_vel"],
                physicsClientId=self.pyb_client,
            )

    @staticmethod
    def motors(drone: Drone, rpms: npt.NDArray[np.float_]) -> list[tuple[int, ForceTorque]]:
        """Base PyBullet physics implementation.

        Args:
            drone: The target drone to calculate the physics for.
            rpms: The rpms to apply to the drone rotors.

        Returns:
            A list of tuples containing the link id and a force/torque tuple.
        """
        ft = []
        forces = np.array(rpms**2) * drone.params.kf
        torques = np.array(rpms**2) * drone.params.km
        z_torque = torques[0] - torques[1] + torques[2] - torques[3]
        for i in range(4):
            ft.append((i, ForceTorque([0, 0, forces[i]], [0, 0, 0])))
        ft.append((4, ForceTorque([0, 0, 0], [0, 0, z_torque])))
        return ft

    def dynamics(self, drone: Drone, rpms: npt.NDArray[np.float_]) -> list[tuple[int, ForceTorque]]:
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Args:
            drone: The target drone to calculate the physics for.
            rpms: The rpm to apply to the drone rotors.
        """
        pos = drone.state["pos"]
        rpy = drone.state["rpy"]
        vel = drone.state["vel"]
        rpy_rates = drone.state["ang_vel"]
        rotation = R.from_euler("XYZ", rpy).as_matrix()
        # Compute forces and torques.
        forces = np.array(rpms**2) * drone.params.kf
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, Constants.GRAVITY])
        z_torques = np.array(rpms**2) * drone.params.km
        z_torque = z_torques[0] - z_torques[1] + z_torques[2] - z_torques[3]
        L = drone.params.arm_len
        x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (L / np.sqrt(2))
        y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (L / np.sqrt(2))
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(drone.params.J, rpy_rates))
        rpy_rates_deriv = np.dot(drone.params.J_inv, torques)
        no_pybullet_dyn_accs = force_world_frame / drone.params.mass
        # Update state.
        vel = vel + no_pybullet_dyn_accs * self.dt
        rpy_rates = rpy_rates + rpy_rates_deriv * self.dt
        drone.state["pos"] = pos + vel * self.dt
        drone.state["rpy"] = rpy + rpy_rates * self.dt
        drone.state["vel"] = vel
        drone.state["ang_vel"] = rpy_rates
        return []  # No forces/torques to apply. We set the drone state directly.

    @staticmethod
    def downwash(drone: Drone, other_drones: list[Drone]) -> list[tuple[int, ForceTorque]]:
        """PyBullet implementation of a ground effect model.

        Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Args:
            drone: The target drone to calculate the downwash effect for.
            other_drones: List of other drones in the environment.

        Returns:
            A list of tuples containing the link id and a force/torque tuple.
        """
        ft = []
        for other_drone in other_drones:
            delta_z = drone.pos[2] - other_drone.pos[2]
            delta_xy = np.linalg.norm(drone.state["pos"][:2] - other_drone.state["pos"][:2])
            if delta_z > 0 and delta_xy < 10:  # Ignore drones more than 10 meters away
                alpha = (
                    other_drone.params.dw_coeff_1
                    * (other_drone.params.prop_radius / (4 * delta_z)) ** 2
                )
                beta = other_drone.params.dw_coeff_2 * delta_z + other_drone.params.dw_coeff_3
                force = [0, 0, -alpha * np.exp(-0.5 * (delta_xy / beta) ** 2)]
                ft.append((4, ForceTorque(force, [0, 0, 0])))
        return ft

    def ground_effect(
        self, drone: Drone, rpms: npt.NDArray[np.float_]
    ) -> list[tuple[int, ForceTorque]]:
        """PyBullet implementation of a ground effect model.

        Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Args:
            drone: The target drone to calculate the ground effect for.
            rpms: The rpms to apply to the drone rotors.

        Returns:
            A list of tuples containing the link id and a force/torque tuple.
        """
        s = p.getLinkStates(
            drone.id,
            linkIndices=[0, 1, 2, 3],
            computeLinkVelocity=1,
            computeForwardKinematics=1,
            physicsClientId=self.pyb_client,
        )
        prop_heights = np.array([s[0][0][2], s[1][0][2], s[2][0][2], s[3][0][2]])
        prop_heights = np.clip(prop_heights, drone.params.gnd_eff_min_height_clip, np.inf)
        gnd_effects = (
            rpms**2
            * drone.params.kf
            * drone.params.gnd_eff_coeff
            * (drone.params.prop_radius / (4 * prop_heights)) ** 2
        )
        ft = []
        if np.abs(drone.state["rpy"][:2]).max() < np.pi / 2:  # Ignore when not approximately level
            for i in range(4):
                ft.append((i, ForceTorque([0, 0, gnd_effects[i]], [0, 0, 0])))
        return ft

    @staticmethod
    def drag(drone: Drone, rpms: npt.NDArray[np.float_]):
        """PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Args:
            drone: The target drone to calculate the ground effect for.
            rpms: The rpms to apply to the drone rotors.

        Returns:
            A list of tuples containing the link id and a force/torque tuple.
        """
        rot = R.from_euler("XYZ", drone.state["rpy"]).as_matrix()
        # Simple draft model applied to the base/center of mass
        drag_factors = -1 * drone.params.drag_coeff * np.sum(np.array(2 * np.pi * rpms / 60))
        drag = np.dot(rot, drag_factors * np.array(drone.state["vel"]))
        return [(4, ForceTorque(drag, [0, 0, 0]))]
