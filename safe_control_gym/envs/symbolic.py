import casadi as cs
from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel
from safe_control_gym.envs.physics import Constants
from safe_control_gym.envs.drone import Drone
from safe_control_gym.math_and_models.transformations import csRotXYZ


def symbolic(drone: Drone, dt: float) -> SymbolicModel:
    """Create symbolic (CasADi) models for dynamics, observation, and cost of a quadcopter.

    Returns:
        The CasADi symbolic model of the environment.
    """
    m, g = drone.nominal_params.mass, Constants.GRAVITY
    # Define states.
    z = cs.MX.sym("z")
    z_dot = cs.MX.sym("z_dot")

    # Set up the dynamics model for a 3D quadrotor.
    nx, nu = 12, 4
    Ixx, Iyy, Izz = drone.nominal_params.J.diagonal()
    J = cs.blockcat([[Ixx, 0.0, 0.0], [0.0, Iyy, 0.0], [0.0, 0.0, Izz]])
    Jinv = cs.blockcat([[1.0 / Ixx, 0.0, 0.0], [0.0, 1.0 / Iyy, 0.0], [0.0, 0.0, 1.0 / Izz]])
    gamma = drone.nominal_params.km / drone.nominal_params.kf
    x = cs.MX.sym("x")
    y = cs.MX.sym("y")
    phi = cs.MX.sym("phi")  # Roll
    theta = cs.MX.sym("theta")  # Pitch
    psi = cs.MX.sym("psi")  # Yaw
    x_dot = cs.MX.sym("x_dot")
    y_dot = cs.MX.sym("y_dot")
    p = cs.MX.sym("p")  # Body frame roll rate
    q = cs.MX.sym("q")  # body frame pith rate
    r = cs.MX.sym("r")  # body frame yaw rate
    # Rotation matrix transforming a vector in the body frame to the world frame. PyBullet Euler
    # angles use the SDFormat for rotation matrices.
    Rob = csRotXYZ(phi, theta, psi)
    # Define state variables.
    X = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r)
    # Define inputs.
    f1 = cs.MX.sym("f1")
    f2 = cs.MX.sym("f2")
    f3 = cs.MX.sym("f3")
    f4 = cs.MX.sym("f4")
    U = cs.vertcat(f1, f2, f3, f4)

    # From Ch. 2 of Luis, Carlos, and Jérôme Le Ny. "Design of a trajectory tracking
    # controller for a nanoquadcopter." arXiv preprint arXiv:1608.05786 (2016).

    # Defining the dynamics function.
    # We are using the velocity of the base wrt to the world frame expressed in the world frame.
    # Note that the reference expresses this in the body frame.
    oVdot_cg_o = Rob @ cs.vertcat(0, 0, f1 + f2 + f3 + f4) / m - cs.vertcat(0, 0, g)
    pos_ddot = oVdot_cg_o
    pos_dot = cs.vertcat(x_dot, y_dot, z_dot)
    Mb = cs.vertcat(
        drone.nominal_params.arm_len / cs.sqrt(2.0) * (f1 + f2 - f3 - f4),
        drone.nominal_params.arm_len / cs.sqrt(2.0) * (-f1 + f2 + f3 - f4),
        gamma * (f1 - f2 + f3 - f4),
    )
    rate_dot = Jinv @ (Mb - (cs.skew(cs.vertcat(p, q, r)) @ J @ cs.vertcat(p, q, r)))
    ang_dot = cs.blockcat(
        [
            [1, cs.sin(phi) * cs.tan(theta), cs.cos(phi) * cs.tan(theta)],
            [0, cs.cos(phi), -cs.sin(phi)],
            [0, cs.sin(phi) / cs.cos(theta), cs.cos(phi) / cs.cos(theta)],
        ]
    ) @ cs.vertcat(p, q, r)
    X_dot = cs.vertcat(
        pos_dot[0],
        pos_ddot[0],
        pos_dot[1],
        pos_ddot[1],
        pos_dot[2],
        pos_ddot[2],
        ang_dot,
        rate_dot,
    )

    Y = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r)

    # Define cost (quadratic form).
    Q = cs.MX.sym("Q", nx, nx)
    R = cs.MX.sym("R", nu, nu)
    Xr = cs.MX.sym("Xr", nx, 1)
    Ur = cs.MX.sym("Ur", nu, 1)
    cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
    # Define dynamics and cost dictionaries.
    dynamics = {"dyn_eqn": X_dot, "obs_eqn": Y, "vars": {"X": X, "U": U}}
    cost = {
        "cost_func": cost_func,
        "vars": {"X": X, "U": U, "Xr": Xr, "Ur": Ur, "Q": Q, "R": R},
    }
    return SymbolicModel(dynamics=dynamics, cost=cost, dt=dt)
