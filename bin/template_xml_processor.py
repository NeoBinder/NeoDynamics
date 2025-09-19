import argparse
from box import Box
import pandas as pd
import numpy as np
from neomd.builder.forcefiled import ComplexForceField
from neomd.builder.ligand import ligands_from_config
from openmm import app, unit
import xml.etree.ElementTree as ET

def generate_template(args):
    config = Box.from_yaml(filename=args.config)
    ligand = ligands_from_config(config.get("ligands"))[0]
    ligand.generate_unique_atom_names()
    modeller = app.Modeller(ligand.molecule.to_topology().to_openmm(),
                            unit.Quantity(ligand.molecule.conformers[0].magnitude,
                                         unit.angstrom)
                            )

    forcefield = ComplexForceField()
    gaff_generator = forcefield.init_gaff_generator()
    gaff_generator.debug_ffxml_filename = config.get("output_xml")
    _res=[res for res in modeller.topology.residues()][0]
    _res.name=ligand.molecule.name
    forcefield.add_molecule(
                            ligand.molecule,
                            gaff_generator,
                            )
    
    if gaff_generator.generator(forcefield.forcefield,_res):
        print(f"Ligand has been successfully parameterized, \
the forcefield parameter has been saved: {gaff_generator.debug_ffxml_filename}.")
    else:
        print(f"Failed to parameterize ligand.")

def fix_torsion_params(torsions,fix_info):
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

    for key,csv_f in fix_info.items():
        class1, class2, class3, class4 = key.split('-')
        torsion = ET.SubElement(torsions, 
                                'Proper',            
                                class1=class1,
                                class2=class2,
                                class3=class3,
                                class4=class4)
        df=pd.read_csv(csv_f)
        for index, row in df.iterrows():
            torsion.set(f'periodicity{index+1}',
                        str(int(np.round(row['periodicity'])))
                        )
            torsion.set(f'phase{index+1}', str(row['phase']))
            torsion.set(f'k{index+1}', str(row['k']))
def strip_all_element_text_tail(root):
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

def modify_template(args):
    config = Box.from_yaml(filename=args.config)
    tree = ET.parse(config.in_xml)
    root = tree.getroot()
    for fix_param,fix_info in config.fix_params.items():
        if fix_param=='torsion':
            fix_torsion_params(root.find('.//PeriodicTorsionForce'),
                               fix_info)
            
    # strip_all_element_text_tail(root)
    pretty_xml_str = prettify_xml(root)
    # tree.write(config.out_xml)
    with open(config.out_xml, "w", encoding="utf-8") as f:
        f.write(pretty_xml_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process templates with .xml format")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # generate_template command
    parser_gen = subparsers.add_parser('generate_template', 
                                    help='对输入的结构生成template.xml')
    parser_gen.add_argument("config", type=str, help="configuration file")
    parser_gen.set_defaults(func=generate_template)

    # modify_template command
    parser_modify = subparsers.add_parser('modify_template', 
                                    help='对已有template.xml更改')
    parser_modify.add_argument("config", type=str, help="configuration file")
    parser_modify.set_defaults(func=modify_template)

    args = parser.parse_args()
    args.func(args)  # 调用对应的函数