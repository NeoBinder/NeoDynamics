"""
Template XML processing.

Two YAML-config-driven subcommands: ``generate_template`` — parameterize
one ligand with GAFF and write the produced residue template (+ additional
parameters) to ``output_xml``; ``modify_template`` — rewrite
``PeriodicTorsionForce`` ``Proper`` torsions of an existing ffxml from CSV
parameter tables (:func:`fix_torsion_params`), then pretty-print
(:func:`prettify_xml`) to ``out_xml`` as utf-8.  Fidelity notes live on the
individual functions.
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
    """The one attribute the generate_template flow needs from a topology
    residue: its (renamed) name — the GAFF generator names the template
    after the ligand.
    """

    def __init__(self, name):
        self.name = name


def generate_template(config, *, runner: ToolRunner | None = None) -> str:
    """Parameterize the (single) configured ligand via GAFF and write the
    ffxml to ``config["output_xml"]``.

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
    # the produced template is written to the configured output file
    with open(output_xml, "w") as outfile:
        outfile.write(ffxml_contents)
    print(f"Ligand has been successfully parameterized, "
          f"the forcefield parameter has been saved: {output_xml}.")
    return ffxml_contents


def fix_torsion_params(torsions, fix_info):
    """Remove every ``Proper`` torsion whose four classes match
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
    """Strip surrounding whitespace off every element's text and tail
    (never called — see the commented call site in
    :func:`modify_template`)."""
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
    """Apply ``config["fix_params"]`` to ``config["in_xml"]`` and
    pretty-print the result to ``config["out_xml"]`` (utf-8)."""
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
    """Read a YAML config file into a plain dict."""
    import yaml

    with open(path) as handle:
        return yaml.safe_load(handle)


def _cli_generate_template(args):
    generate_template(_load_config(args.config))


def _cli_modify_template(args):
    modify_template(_load_config(args.config))


def main(argv=None):
    """The template_xml_processor CLI; ``argv`` defaults to
    ``sys.argv[1:]``."""
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
