"""Template XML processing — verbatim port of v1 ``bin/template_xml_processor.py``
(v2 migration plan §6 parity row "Template XML processing"; verification =
ffxml hash).

v1's standalone script had two subcommands, each driven by a YAML config file
(v1 loaded it with ``Box.from_yaml``; the port reads the same file with
``yaml.safe_load`` into the plain-dict config convention used across neomd):

``generate_template``
    Parameterize one ligand with GAFF and write the produced residue
    template (+ additional parameters) to ``output_xml``.  v1 built an
    openmm ``Modeller`` from the ligand, renamed its first residue to
    ``ligand.molecule.name``, registered a ``ComplexForceField`` GAFF
    generator with ``debug_ffxml_filename = output_xml`` — the debug-file
    trick WAS the output mechanism — and let openmm's template matching
    call ``generator(forcefield.forcefield, residue)``.  The v2 route calls
    :class:`~neomd.tools.antechamber.AntechamberBackend.generate_residue_template`
    directly (the same parameterization knowledge, without openmm
    ``ForceField`` scaffolding just to harvest the xml); the returned
    ffxml string is written to ``config["output_xml"]`` — v1's
    debug-file mechanism, relocated to the caller.

``modify_template``
    Rewrite ``PeriodicTorsionForce`` ``Proper`` torsions of an existing
    ffxml from CSV parameter tables (:func:`fix_torsion_params`, verbatim:
    both-direction class-key matching, removal loop, rebuild loop with
    periodicity/phase/k columns and an optional ``divide_factor`` the k
    column is divided by), then pretty-print (:func:`prettify_xml`) and
    write ``out_xml`` as utf-8.

Fidelity notes (deviations, all deliberate):

* v1's ``generate_template`` printed its failure message when openmm's
  generator callback returned falsy.  The direct backend route cannot
  return falsy — failure surfaces as an exception — so
  :func:`generate_template` prints v1's ``Failed to parameterize
  ligand.`` and re-raises (v1 itself let antechamber exceptions
  propagate as tracebacks; only the falsy branch was silent).
* ``fix_torsion_params`` skipped fix entries whose value was not a
  ``Box``; the dict port checks ``isinstance(..., dict)`` — the same
  filter with the plain config type.
* v1's ``modify_template`` contains a COMMENTED-OUT call to
  ``strip_all_element_text_tail(root)``; the comment (and the whitespace
  behavior it implies — element text/tail is NOT stripped before
  pretty-printing) is kept verbatim.  See :func:`modify_template`.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from neomd.tools.antechamber import AntechamberBackend
from neomd.tools.ligand import ligands_from_config
from neomd.tools.port import SubprocessToolRunner, ToolRunner

__all__ = [
    "generate_template",
    "fix_torsion_params",
    "strip_all_element_text_tail",
    "prettify_xml",
    "modify_template",
    "main",
]


class _Residue:
    """The one attribute of an openmm topology residue that v1's
    ``generate_template`` flow consumed: its (renamed) name.  v1 built a
    whole ``app.Modeller`` and renamed its first residue to
    ``ligand.molecule.name`` so the GAFF generator would name the template
    after the ligand; the backend route needs only that name.
    """

    def __init__(self, name):
        self.name = name


def generate_template(config, *, runner: ToolRunner | None = None) -> str:
    """v1 ``generate_template`` port: parameterize the (single) configured
    ligand via GAFF and write the ffxml to ``config["output_xml"]``.

    ``runner`` selects the :class:`~neomd.tools.port.ToolRunner` executing
    antechamber/parmchk2 (a ``FakeToolRunner`` in tests; a
    :class:`~neomd.tools.port.SubprocessToolRunner` by default).  Returns
    the ffxml contents written.
    """
    ligand = ligands_from_config(config["ligands"])[0]
    ligand.generate_unique_atom_names()
    output_xml = config["output_xml"]
    backend = AntechamberBackend(
        runner if runner is not None else SubprocessToolRunner())
    try:
        ffxml_contents = backend.generate_residue_template(
            ligand.molecule, original_residue=_Residue(ligand.molecule.name))
    except Exception:
        print("Failed to parameterize ligand.")
        raise
    # v1's debug_ffxml_filename mechanism: the produced template is written
    # to the configured output file
    with open(output_xml, "w") as outfile:
        outfile.write(ffxml_contents)
    print(f"Ligand has been successfully parameterized, "
          f"the forcefield parameter has been saved: {output_xml}.")
    return ffxml_contents


def fix_torsion_params(torsions, fix_info):
    """v1 verbatim: remove every ``Proper`` torsion whose four classes match
    a fix key (in either direction, ``c1-c2-c3-c4`` or reversed), then
    rebuild one ``Proper`` per keyed fix entry from its CSV table
    (``periodicity`` rounded to int, ``phase`` verbatim, ``k`` divided by
    ``divide_factor``, default 1).
    """
    for torsion in torsions.findall('Proper'):
        class1 = torsion.get('class1')
        class2 = torsion.get('class2')
        class3 = torsion.get('class3')
        class4 = torsion.get('class4')
        key1 = f"{class1}-{class2}-{class3}-{class4}"
        key2 = f"{class4}-{class3}-{class2}-{class1}"
        if key1 in fix_info.keys():
            key = key1
        elif key2 in fix_info.keys():
            key = key2
        else:
            continue
        torsions.remove(torsion)

    for key, _fix_info in fix_info.items():
        # v1 skipped entries that were not Box objects; the dict port
        if not isinstance(_fix_info, dict):
            continue
        if not _fix_info.get('param_csv'):
            continue
        class1, class2, class3, class4 = key.split('-')
        torsion = ET.SubElement(torsions,
                                'Proper',
                                class1=class1,
                                class2=class2,
                                class3=class3,
                                class4=class4)
        divide_factor = _fix_info.get('divide_factor', 1)
        df = pd.read_csv(_fix_info['param_csv'])
        for index, row in df.iterrows():
            torsion.set(f'periodicity{index+1}',
                        str(int(np.round(row['periodicity'])))
                        )
            torsion.set(f'phase{index+1}', str(row['phase']))
            torsion.set(f'k{index+1}',
                        str(row['k']/divide_factor)
                        )


def strip_all_element_text_tail(root):
    """v1 verbatim (and, exactly as in v1, never called — see the commented
    call site in :func:`modify_template`): strip surrounding whitespace off
    every element's text and tail."""
    for e in root.iter():
        if e.text is not None:
            original = e.text
            stripped = original.strip()
            if stripped:
                e.text = stripped
            else:
                e.text = None
        if e.tail is not None:
            original = e.tail
            stripped = original.strip()
            if stripped:
                e.tail = stripped
            else:
                e.tail = None


def prettify_xml(elem):
    """
    将 Element 元素转换为带缩进的美观 XML 字符串

    参数:
        elem: XML 元素对象

    返回:
        格式化后的 XML 字符串
    """
    from xml.dom import minidom
    # 将元素转换为字符串
    rough_string = ET.tostring(elem, 'utf-8')
    # 解析字符串为 DOM 对象
    reparsed = minidom.parseString(rough_string)
    # 生成带缩进的 XML（indent 控制缩进空格数）
    pretty_xml_str = reparsed.toprettyxml(indent="  ")
    pretty_xml_str = '\n'.join([line for line in pretty_xml_str.split('\n') if line.strip()])
    return pretty_xml_str


def modify_template(config) -> None:
    """v1 ``modify_template`` port: apply ``config["fix_params"]`` to
    ``config["in_xml"]`` and pretty-print the result to
    ``config["out_xml"]`` (utf-8)."""
    tree = ET.parse(config["in_xml"])
    root = tree.getroot()
    for fix_param, fix_info in config["fix_params"].items():
        if fix_param == 'torsion':
            fix_torsion_params(root.find('.//PeriodicTorsionForce'),
                               fix_info)

    # strip_all_element_text_tail(root)
    pretty_xml_str = prettify_xml(root)
    # tree.write(config.out_xml)
    with open(config["out_xml"], "w", encoding="utf-8") as f:
        f.write(pretty_xml_str)


def _load_config(path):
    """v1's ``Box.from_yaml(filename=...)`` as the neomd plain-dict read."""
    import yaml

    with open(path) as handle:
        return yaml.safe_load(handle)


def _cli_generate_template(args):
    generate_template(_load_config(args.config))


def _cli_modify_template(args):
    modify_template(_load_config(args.config))


def main(argv=None):
    """The v1 template_xml_processor CLI (argparse surface verbatim);
    ``argv`` defaults to ``sys.argv[1:]``."""
    parser = argparse.ArgumentParser(description="process templates with .xml format")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # generate_template command
    parser_gen = subparsers.add_parser('generate_template',
                                       help='对输入的结构生成template.xml')
    parser_gen.add_argument("config", type=str, help="configuration file")
    parser_gen.set_defaults(func=_cli_generate_template)

    # modify_template command
    parser_modify = subparsers.add_parser('modify_template',
                                          help='对已有template.xml更改')
    parser_modify.add_argument("config", type=str, help="configuration file")
    parser_modify.set_defaults(func=_cli_modify_template)

    args = parser.parse_args(argv)
    args.func(args)  # 调用对应的函数


if __name__ == "__main__":  # pragma: no cover
    main()
