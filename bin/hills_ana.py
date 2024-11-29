import math
import os

import mdtraj
import numpy as np
import pandas as pd
from MDAnalysis.lib.formats.libdcd import DCDFile
from openmm.app.pdbfile import PDBFile
from ttk.calculator import get_center_of_mass
from ttk.io import topology_parser


def w_or_a(filename):
    if os.path.exists(filename):
        append_write = "a"  # append if already exists
    else:
        append_write = "w"  # make a new file if not
    return append_write


def df_from_hills(hill_path):
    f = open(hill_path, "r")
    hills = f.readlines()
    f.close()
    head_line = hills[0].split()[2:]
    hills_table = np.transpose([np.float_(x.split()) for x in hills[3:]])
    hills_dic = {head_line[i]: hills_table[i] for i in range(len(head_line))}
    hills_df = pd.core.frame.DataFrame(hills_dic)
    return hills_df


def time_from_cvs(hill_path, cv_values, cv_names, cv_buffer=0.1):
    df = df_from_hills(hill_path)
    if len(cv_names) == 3:
        df_sele = df[
            (df[cv_names[0]] > cv_values[0] - cv_buffer)
            & (df[cv_names[0]] < cv_values[0] + cv_buffer)
            & (df[cv_names[1]] > cv_values[1] - cv_buffer)
            & (df[cv_names[1]] < cv_values[1] + cv_buffer)
            & (df[cv_names[2]] > cv_values[2] - cv_buffer)
            & (df[cv_names[2]] < cv_values[2] + cv_buffer)
        ]
    elif len(cv_names) == 2:
        df_sele = df[
            (df[cv_names[0]] > cv_values[0] - cv_buffer)
            & (df[cv_names[0]] < cv_values[0] + cv_buffer)
            & (df[cv_names[1]] > cv_values[1] - cv_buffer)
            & (df[cv_names[1]] < cv_values[1] + cv_buffer)
        ]
    return df_sele[["time"]]


def get_traj(dcdpath):
    dcdf = DCDFile(dcdpath)
    for frame in dcdf:
        yield frame.xyz


def get_dihedral(p, in_degree=False):
    # p=np.array([p1,p2,p3,p4])
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array([v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
    # Normalize vectors
    v /= np.sqrt(np.einsum("...i,...i", v, v)).reshape(-1, 1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    if in_degree:
        return np.degrees(np.arctan2(y, x))
    else:
        return np.arctan2(y, x)


def pdb_from_dcd_cvs(
    dcd_f,
    in_top_f,
    out_path,
    cv_type="distance",
    cv_ids=None,
    cv_ref=None,
    tolers=None,
    out_split=False,
    out_top_f=None,
):
    if not out_top_f is None:
        out_top = PDBFile(out_top_f).getTopology()
    else:
        out_top = PDBFile(top_f).getTopology()
    if top_f.endswith(".pdb"):
        ttk_top = topology_parser.topology_from_pdb(top_f)
    elif top_f.endswith(".pdbx"):
        from openmm.app.pdbxfile import PDBxFile

        opmm_top = PDBxFile(top_f).getTopology()
        ttk_top = topology_parser.topology_from_openmmm(opmm_top)
    frames = []
    first_sele = 0
    for frame_id, frame in enumerate(get_traj(dcd_f)):
        ttk_top.positions = frame
        is_sele = True
        if cv_type == "distance":
            for cv_nums in range(len(cv_ids)):
                grp1 = cv_ids[cv_nums][0]
                grp2 = cv_ids[cv_nums][1]
                com1 = get_center_of_mass(ttk_top.get_atom_by_indices(grp1))
                com2 = get_center_of_mass(ttk_top.get_atom_by_indices(grp2))
                dist = math.dist(com1, com2)

                if isinstance(cv_ref[cv_nums], list) and isinstance(
                    tolers[cv_nums], list
                ):
                    for i in range(len(cv_ref[cv_nums])):
                        _cv_ref = cv_ref[cv_nums][i]
                        _toler = tolers[cv_nums][i]
                        is_sele = is_sele and (abs(dist - _cv_ref) < _toler)
                elif isinstance(cv_ref[cv_nums], float) and isinstance(
                    tolers[cv_nums], float
                ):
                    is_sele = is_sele and (
                        abs(dist - cv_ref[cv_nums]) < tolers[cv_nums]
                    )
                else:
                    raise AssertionError
                # print('{} {}'.format(frame_id,dist))
        elif cv_type == "torsion":
            for cv_nums in range(len(cv_ids)):
                ids = cv_ids[cv_nums]
                positions = []
                for _id in ids:
                    _atom = ttk_top.get_atom_by_indices([_id])[0]
                    _pos = _atom.position
                    positions.append(_pos)
                dihedral = get_dihedral(np.array(positions), in_degree=False)

                if isinstance(cv_ref[cv_nums], list) and isinstance(
                    tolers[cv_nums], list
                ):
                    _ls = [
                        (abs(dihedral - cv_ref[cv_nums][i]) < tolers[cv_nums][i])
                        for i in range(len(cv_ref[cv_nums]))
                    ]
                    is_sele = (True in _ls) and is_sele
                    print("frame{}: {},{}".format(frame_id, is_sele, dihedral))
                elif isinstance(cv_ref[cv_nums], float) and isinstance(
                    tolers[cv_nums], float
                ):
                    is_sele = is_sele and (
                        abs(dihedral - cv_ref[cv_nums]) < tolers[cv_nums]
                    )
                else:
                    raise AssertionError
                # print('{} {}'.format(frame_id,dihedral))
        else:
            raise NotImplementedError("cv_type:{} not implemented".format(cv_type))
        if is_sele:
            if out_split:
                print("select frame {}\n".format(frame_id))
                out_f = os.path.join(out_path, str(frame_id) + ".pdb")
                PDBFile.writeFile(out_top, frame, file=(open(out_f, "w")))
            else:
                out_f = os.path.join(out_path, "out_select.pdb")
                if first_sele == 0:
                    wa = "w"
                    first_sele += 1
                else:
                    wa = "a"
                PDBFile.writeFile(out_top, frame, file=(open(out_f, wa)))


"""
pdb_from_dcd_cvs(dcd_f,top_f,out_path,
                      cv_type='torsion',
                      cv_ids=cv_ids,
                      cv_ref=[[np.pi,-np.pi],[np.pi,-np.pi]],
                      tolers=[[0.1,0,1],[0.1,0.1]],
                      out_split=True,
                      out_top_f='/export/hanxinhao/project/meta_config/3qqa/min_eq_min_3/min_eq_min.pdb'
                      )
"""


def calc_com_trj(trj, idx):
    sele_trj = trj.atom_slice(idx)
    cent = mdtraj.compute_center_of_geometry(sele_trj)
    return cent


def calc_dist_com_trj(pdb_f, idx1, idx2, top=None):
    if trj_f.endswith(".pdb"):
        trj = mdtraj.load_pdb(trj_f)
    elif trj_f.endswith(".dcd"):
        trj = mdtraj.load_dcd(trj_f, top=top)
    cent1 = calc_com_trj(trj, idx1)
    cent2 = calc_com_trj(trj, idx2)

    distance = []
    for i in range(len(cent1)):
        distance.append(math.dist(cent1[i], cent2[i]))
    return distance


"""
calc_dist_com_trj(
'../../min/last.pdb',
[5424,5455,5793,5808,5859,6047,6064],
[6608,6609,6610,6611,6612,6613,6614,6615,6616],
)
"""
