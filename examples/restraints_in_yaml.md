This file contains format/keywords and explains of restarints that can be used in the YAML file.
```python
    if restraint_config["type"] == "sphere":
        restraint = generate_restraint_sphere(restraint_config)
    elif restraint_config["type"] == "funnel":
        restraint = generate_restraint_funnel(restraint_config)
    elif restraint_config["type"] == "distance":
        restraint = generate_restraint_distance(restraint_config)
    elif restraint_config["type"] == "angle":
        restraint = generate_restraint_angle(restraint_config)
    elif restraint_config["type"] == "dihedral":
        restraint = generate_restraint_dihedral(restraint_config)
    elif restraint_config["type"] == "ref_file":
        restraint = generate_restraint_ref_file(restraint_config, system=system)
    elif restraint_config["type"] == "dist_ref_position":
        restraint = generate_dist_ref_position(restraint_config)
    elif restraint_config["type"] == "xyz_box":
        restraint = generate_xyz_box(restraint_config)
    elif restraint_config["type"] == "vec_restraint":
        restraint = generate_vec_restraint(restraint_config)
    elif restraint_config["type"] == "test":
        restraint = force_test(restraint_config)
```
