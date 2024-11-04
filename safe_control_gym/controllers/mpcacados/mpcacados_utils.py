import casadi as ca
import numpy as np

def detect_constr(model, constraints, stage_type):
    x = model.x
    u = model.u
    z = model.z
    nx = x.shape[0]
    nu = u.shape[0]
    # nz = z.shape[0] # TODO

    if isinstance(x, ca.SX):
        isSX = True
    else:
        raise ValueError('Constraint detection only works for casadi.SX!')

    if stage_type == 'initial':
        expr_constr = model.con_h_expr_0
        LB = constraints.lh_0
        UB = constraints.uh_0
        print('\nConstraint detection for initial constraints.')
    elif stage_type == 'path':
        expr_constr = model.con_h_expr
        LB = constraints.lh
        UB = constraints.uh
        print('\nConstraint detection for path constraints.')
    elif stage_type == 'terminal':
        expr_constr = model.con_h_expr_e
        LB = constraints.lh_e
        UB = constraints.uh_e
        print('\nConstraint detection for terminal constraints.')
    else:
        raise ValueError('Constraint detection: Wrong stage_type.')

    if expr_constr is None:
        expr_constr = ca.SX.sym('con_h_expr', 0, 0)

    if not isinstance(expr_constr, (ca.SX, ca.SX)):
        print('expr_constr =', expr_constr)
        raise ValueError("Constraint type detection requires definition of constraints as CasADi SX or SX.")

    # Initialize
    constr_expr_h = ca.SX.sym('con_h_expr', 0, 0)
    lh = []
    uh = []

    C = ca.SX.zeros(0, nx)
    D = ca.SX.zeros(0, nu)
    lg = []
    ug = []

    Jbx = ca.SX.zeros(0, nx)
    lbx = []
    ubx = []

    Jbu = ca.SX.zeros(0, nu)
    lbu = []
    ubu = []

    # Loop over CasADi formulated constraints
    for ii in range(expr_constr.shape[0]):
        c = expr_constr[ii]
        if any(ca.which_depends(c, z)) or not ca.is_linear(c, ca.vertcat(x, u)) or any(ca.which_depends(c, model.p)): # TODO or any(ca.which_depends(c, model.p_global)):
            # External constraint
            constr_expr_h = ca.vertcat(constr_expr_h, c)
            lh.append(LB[ii])
            uh.append(UB[ii])
            print(f'constraint {ii+1} is kept as nonlinear constraint.')
            print(c)
            print(' ')
        else:  # c is linear in x and u
            Jc_fun = ca.Function('Jc_fun', [x[0]], [ca.jacobian(c, ca.vertcat(x, u))])
            Jc = Jc_fun(0).full().squeeze()

            if np.sum(Jc != 0) == 1:
                # c is bound
                idb = Jc.nonzero()[0][0]
                if idb < nx:
                    # Bound on x
                    Jbx = ca.vertcat(Jbx, ca.SX.zeros(1, nx))
                    Jbx[-1, idb] = 1
                    lbx.append(LB[ii] / Jc[idb])
                    ubx.append(UB[ii] / Jc[idb])
                    print(f'constraint {ii+1} is reformulated as bound on x.')
                    print(c)
                    print(' ')
                else:
                    # Bound on u
                    Jbu = ca.vertcat(Jbu, ca.SX.zeros(1, nu))
                    Jbu[-1, idb - nx] = 1
                    lbu.append(LB[ii] / Jc[idb])
                    ubu.append(UB[ii] / Jc[idb])
                    print(f'constraint {ii+1} is reformulated as bound on u.')
                    print(c)
                    print(' ')
            else:
                # c is general linear constraint
                C = ca.vertcat(C, Jc[0:nx])
                D = ca.vertcat(D, Jc[nx:])
                lg.append(LB[ii])
                ug.append(UB[ii])
                print(f'constraint {ii+1} is reformulated as general linear constraint.')
                print(c)
                print(' ')

    if stage_type == 'terminal':
        # Checks
        if any(ca.which_depends(expr_constr, u)) or lbu or (D.size()[0] > 0 and any(D)):
            raise ValueError('Terminal constraint may not depend on control input.')
        # h
        constraints.constr_type_e = 'BGH'
        if lh:
            model.con_h_expr_e = constr_expr_h
            constraints.lh_e = np.array(lh)
            constraints.uh_e = np.array(uh)
        else:
            model.con_h_expr_e = None
            constraints.lh_e = np.array([])
            constraints.uh_e = np.array([])
        # g
        if lg:
            constraints.C_e = C
            constraints.lg_e = np.array(lg)
            constraints.ug_e = np.array(ug)
        # Bounds x
        if lbx:
            constraints.idxbx_e = J_to_idx(Jbx)
            constraints.lbx_e = np.array(lbx)
            constraints.ubx_e = np.array(ubx)

    elif stage_type == 'initial':
        print("At initial stage, only h constraints are detected.")
        constraints.constr_type_0 = 'BGH'
        # h
        if lh:
            model.con_h_expr_0 = constr_expr_h
            constraints.lh_0 = np.array(lh)
            constraints.uh_0 = np.array(uh)
        else:
            model.con_h_expr_0 = None
            constraints.lh_0 = np.array([])
            constraints.uh_0 = np.array([])
    else:  # path
        constraints.constr_type = 'BGH'
        # h
        if lh:
            model.con_h_expr = constr_expr_h
            constraints.lh = np.array(lh)
            constraints.uh = np.array(uh)
        else:
            model.con_h_expr = None
            constraints.lh = np.array([])
            constraints.uh = np.array([])
        if lg:
            constraints.C = C
            constraints.D = D
            constraints.lg = np.array(lg)
            constraints.ug = np.array(ug)
        # Bounds x
        if lbx:
            constraints.idxbx = J_to_idx(Jbx)
            constraints.lbx = np.array(lbx)
            constraints.ubx = np.array(ubx)
        # Bounds u
        if lbu:
            constraints.idxbu = J_to_idx(Jbu)
            constraints.lbu = np.array(lbu)
            constraints.ubu = np.array(ubu)
        # g
        if lg:
            model.constr_C = C
            model.constr_D = D
            constraints.lg = np.array(lg)
            constraints.ug = np.array(ug)

def J_to_idx(J):
    nrows = J.size()[0]
    idx = []
    for i in range(nrows):
        this_idx = ca.DM(J[i, :]).full().squeeze().nonzero()[0]
        if len(this_idx) != 1:
            raise ValueError(f'J_to_idx: Invalid J matrix. Exiting. Found more than one nonzero in row {i+1}.')
        if J[i, this_idx] != 1:
            raise ValueError(f'J_to_idx: J matrices can only contain 1s, got J({i+1}, {this_idx}) = {J[i, this_idx]}')
        idx.append(this_idx[0])  # store 0-based index
    return np.array(idx)
