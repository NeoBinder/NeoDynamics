import argparse
import subprocess
import os
import sys
import shutil

class RESP:
    def __init__(self, input_file, charge=0, multiplicity=1, solvent='Water', delta=0.5,
                 orca_path='orca',
                 orca_2mkl_path='orca_2mkl',
                 nprocs=8, maxcore=1000,
                 keyword='! B3LYP/G D3 def2-TZVP def2/J RIJCOSX',
                    output_file='resp2.chg',
                    equivcon=False):
        """
        初始化RESP计算类
        
        Args:
            input_file (str): 输入分子结构文件(.xyz)的路径
            charge (int): 分子净电荷
            multiplicity (int): 自旋多重度
            solvent (str): 溶剂名称
            delta (float): RESP2液相charge权重系数
            orca_path (str): ORCA程序路径
            orca_2mkl_path (str): orca_2mkl工具路径
            nprocs (int): CPU核心数
            maxcore (int): 每个核心内存(MB)
            keyword (str): ORCA计算关键词
        """
        self.input_file = input_file
        self.charge = charge
        self.multiplicity = multiplicity
        self.solvent = solvent
        self.delta = delta
        self.orca_path = orca_path
        self.orca_2mkl_path = orca_2mkl_path
        self.nprocs = nprocs
        self.maxcore = maxcore
        self.keyword = keyword
        self.output_file = output_file
        self.input_equivcon = equivcon
        self.total_equivcon = None
        
        # 获取输入文件的目录和基本名称
        self.work_dir = os.path.dirname(os.path.abspath(input_file)) if not os.path.dirname(input_file) else '.'
        self.base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # 设置所有文件路径
        self._set_file_paths()
    
    def _set_file_paths(self):
        """
        设置所有文件的路径
        """
        # 输入文件
        self.xyz_file = self.input_file
        
        # 临时和中间文件
        self.nval_file = os.path.join(self.work_dir, 'Nval.txt')
        self.gas_inp_file = os.path.join(self.work_dir, 'gas.inp')
        self.gas_out_file = os.path.join(self.work_dir, 'gas.out')
        self.solv_inp_file = os.path.join(self.work_dir, 'solv.inp')
        self.solv_out_file = os.path.join(self.work_dir, 'solv.out')
        self.gas_molden_file = os.path.join(self.work_dir, 'gas.molden')
        self.solv_molden_file = os.path.join(self.work_dir, 'solv.molden')
        self.gas_chg_file = os.path.join(self.work_dir, 'gas.chg')
        self.solv_chg_file = os.path.join(self.work_dir, 'solv.chg')
        if self.input_equivcon:
            self.total_equivcon=os.path.join(self.work_dir, 'eqvcons.txt')
        
        # ORCA生成的其他文件
        self.gas_gbw_file = os.path.join(self.work_dir, 'gas.gbw')
        self.solv_gbw_file = os.path.join(self.work_dir, 'solv.gbw')
        self.gas_prop_file = os.path.join(self.work_dir, 'gas.prop')
        self.solv_prop_file = os.path.join(self.work_dir, 'solv.prop')
        self.gas_molden_input = os.path.join(self.work_dir, 'gas.molden.input')
        self.solv_molden_input = os.path.join(self.work_dir, 'solv.molden.input')
    
    def create_nval_file(self):
        """
        创建价电子数定义文件
        """
        nval_content = """[Nval]
Rb  9
Sr 10
Y  11
Zr 12
Nb 13
Mo 14
Tc 15
Ru 16
Rh 17
Pd 18
Ag 19
Cd 20
In 21
Sn 22
Sb 23
Te 24
I  25
Xe 26
Cs  9
Ba 10
La 11
Ce 30
Pr 31
Nd 32
Pm 33
Sm 34
Eu 35
Gd 36
Tb 37
Dy 38
Ho 39
Er 40
Tm 41
Yb 42
Lu 43
Hf 12
Ta 13
W  14
Re 15
Os 16
Ir 17
Pt 18
Au 19
Hg 20
Tl 21
Pb 22
Bi 23
Po 24
At 25
Rn 26
"""
        with open(self.nval_file, 'w') as f:
            f.write(nval_content)
    
    def create_orca_input(self, input_file, charge, multiplicity, xyz_file, solvent=None):
        """
        创建ORCA输入文件
        
        Args:
            input_file (str): 输入文件路径
            charge (int): 电荷
            multiplicity (int): 自旋多重度
            xyz_file (str): XYZ坐标文件路径
            solvent (str, optional): 溶剂名称
        """
        with open(input_file, 'w') as f:
            f.write(f"{self.keyword}\n")
            f.write(f"%maxcore {self.maxcore}\n")
            f.write(f"%pal nprocs {self.nprocs} end\n")
            
            if solvent:
                f.write("%cpcm\n")
                f.write("smd true\n")
                f.write(f'SMDsolvent "{solvent}"\n')
                f.write("end\n")
            
            f.write(f"* xyz {charge} {multiplicity}\n")
        
        # 添加坐标信息
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
            with open(input_file, 'a') as inp_file:
                for line in lines[2:]:  # 跳过前两行
                    inp_file.write(line)
                inp_file.write("*\n")
    
    def run_orca(self, input_file, output_file):
        """
        运行ORCA计算
        
        Args:
            input_file (str): 输入文件路径
            output_file (str): 输出文件路径
            
        Returns:
            bool: 计算是否成功
        """
        print(f"正在运行ORCA计算: {input_file}")
        orca_cmd = [self.orca_path, input_file]
        
        with open(output_file, 'w') as f:
            return_code = subprocess.call(orca_cmd, stdout=f, stderr=subprocess.STDOUT)
        
        # 检查是否正常结束
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                content = f.read()
                if "ORCA TERMINATED NORMALLY" in content:
                    print("ORCA计算完成!")
                    return True
                else:
                    print(f"ORCA计算失败! 请检查 {output_file} 文件")
                    return False
        else:
            print(f"ORCA计算失败! {output_file} 文件不存在")
            return False
    
    def convert_to_molden(self, input_name, molden_file):
        """
        将ORCA结果转换为Molden格式
        
        Args:
            input_name (str): ORCA输入文件的基本名称(不包含扩展名)
            molden_file (str): 输出的Molden文件路径
            
        Returns:
            bool: 转换是否成功
        """
        print(f"正在转换为Molden格式: {input_name}")
        try:
            # 构建完整的路径
            orca_2mkl_cmd = [self.orca_2mkl_path, os.path.join(self.work_dir, input_name), '-molden']
            subprocess.run(orca_2mkl_cmd, 
                          cwd=self.work_dir,
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL)
            
            # 检查中间文件是否存在
            molden_input_file = os.path.join(self.work_dir, f"{input_name}.molden.input")
            if not os.path.exists(molden_input_file):
                print(f"转换失败: {molden_input_file} 不存在")
                return False
            
            # 合并Nval.txt和.molden.input
            with open(self.nval_file, 'r') as nval_file:
                nval_content = nval_file.read()
            
            with open(molden_input_file, 'r') as molden_file_input:
                molden_content = molden_file_input.read()
            
            with open(molden_file, 'w') as output_file:
                output_file.write(nval_content)
                output_file.write(molden_content)
                
            return True
        except Exception as e:
            print(f"转换Molden格式时出错: {e}")
            return False
    
    def run_multiwfn_resp(self, molden_file, chg_file):
        """
        使用Multiwfn计算RESP电荷
        
        Args:
            molden_file (str): Molden文件路径
            chg_file (str): 输出电荷文件路径
            
        Returns:
            bool: 计算是否成功
        """
        print(f"正在运行Multiwfn计算RESP电荷: {molden_file}")
        try:
            process = subprocess.Popen(
                ['Multiwfn', molden_file, '-ispecial', '1'],
                cwd=self.work_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True
            )
            if self.input_equivcon:
                commands = "7\n18\n5\n1\n\n1\ny\n0\n0\nq\n"
            else:
                commands = "7\n18\n1\ny\n0\n0\nq\n"
            process.communicate(input=commands)
            
            # 检查电荷文件是否生成
            if process.returncode == 0 and os.path.exists(chg_file):
                return True
            else:
                return False
        except Exception as e:
            print(f"Multiwfn计算出错: {e}")
            return False
    
    @staticmethod
    def _read_chg(path):
        """
        读取.chg电荷文件,返回每行拆分后的列表
        """
        charges = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    charges.append(line.strip().split())
        return charges

    def calculate_resp2_charges(self):
        """
        计算RESP2电荷
        """
        # 读取气相和溶剂相电荷
        gas_charges = self._read_chg(self.gas_chg_file)
        solv_charges = self._read_chg(self.solv_chg_file)
        
        # 计算RESP2电荷
        with open(self.output_file, 'w') as f:
            for i in range(len(gas_charges)):
                atom_info = gas_charges[i][0:4]
                gas_charge = float(gas_charges[i][4])
                solv_charge = float(solv_charges[i][4])
                resp2_charge = (1 - self.delta) * gas_charge + self.delta * solv_charge
                
                f.write(f"{atom_info[0]:<3s} {float(atom_info[1]):12.6f} {float(atom_info[2]):12.6f} {float(atom_info[3]):12.6f} {resp2_charge:15.10f}\n")
        
        print(f"计算完成! RESP2电荷已导出到 {self.output_file}")
    
    def cleanup_temp_files(self):
        """
        清理临时文件
        """
        temp_files = [
            self.nval_file,
            self.gas_inp_file,
            self.gas_out_file,
            self.solv_inp_file,
            self.solv_out_file,
            self.gas_molden_input,
            self.solv_molden_input,
            self.gas_gbw_file,
            self.solv_gbw_file,
            self.gas_prop_file,
            self.solv_prop_file
        ]
        
        for file in temp_files:
            if os.path.exists(file):
                try:
                    if os.path.isdir(file):
                        shutil.rmtree(file)
                    else:
                        os.remove(file)
                except OSError:
                    pass
    
    def generate_equivcon_file(self):
        """
        生成等价电荷限制文件
        """
        try:
            with open(self.input_equivcon, 'r') as f:
                lines=f.readlines()
            equivcons=[[x.strip() for x in line.split(',')] for line in lines]
            process = subprocess.Popen(
                ['Multiwfn', self.gas_molden_file, '-ispecial', '1'],
                cwd=self.work_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True
            )
            commands = "7\n18\n5\n10\n0\n0\n0\nq\n"
            process.communicate(input=commands)
            eqvcons_H=os.path.join(self.work_dir, 'eqvcons_H.txt')
            with open(eqvcons_H, 'r') as f:
                lines=f.readlines()
            equivcons = equivcons + [[x.strip() for x in line.split(',')] for line in lines]
            equivcons = [[f'{x:>6}' for x in line] for line in equivcons]
            with open(self.total_equivcon, 'w') as f:
                for equivcon in equivcons:
                    f.write(','.join(equivcon) + '\n')
            print(f"已生成等价电荷限制文件: {self.total_equivcon}")
        except Exception as e:
            print(f"生成等价电荷限制出错: {e}")
            return False

    def run(self, cleanup=True):
        """
        运行完整的RESP2计算
        
        Args:
            cleanup (bool): 是否清理临时文件
        """
        # 检查输入文件是否存在
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"输入文件 {self.input_file} 不存在!")
        
        # 步骤1: 创建Nval.txt文件
        self.create_nval_file()
        
        # 步骤2: 气相计算
        print("\n=== 气相单点计算 ===")
        self.create_orca_input(self.gas_inp_file, self.charge, self.multiplicity, self.xyz_file)
        if not self.run_orca(self.gas_inp_file, self.gas_out_file):
            raise RuntimeError("气相ORCA计算失败!")
        
        if not self.convert_to_molden('gas', self.gas_molden_file):
            raise RuntimeError("转换气相结果为Molden格式失败!")

        if self.input_equivcon: self.generate_equivcon_file()
        if not self.run_multiwfn_resp(self.gas_molden_file, self.gas_chg_file):
            raise RuntimeError("气相RESP电荷计算失败!")

        print(f"气相RESP电荷已保存到 {self.gas_chg_file}")
        
        # 步骤3: 溶剂相计算
        print("\n=== 溶剂相单点计算 ===")
        self.create_orca_input(self.solv_inp_file, self.charge, self.multiplicity, self.xyz_file,
                                solvent=self.solvent)
        
        if not self.run_orca(self.solv_inp_file, self.solv_out_file):
            raise RuntimeError("溶剂相ORCA计算失败!")
        
        if not self.convert_to_molden('solv', self.solv_molden_file):
            raise RuntimeError("转换溶剂相结果为Molden格式失败!")

        if not self.run_multiwfn_resp(self.solv_molden_file, self.solv_chg_file):
            raise RuntimeError("溶剂相RESP电荷计算失败!")
        
        print(f"溶剂相RESP电荷已保存到 {self.solv_chg_file}")
        
        # 步骤4: 计算RESP2电荷
        print("\n=== 计算RESP2电荷 ===")
        self.calculate_resp2_charges()
        
        # 清理临时文件
        if cleanup:
            self.cleanup_temp_files()
        
        print("\n所有计算完成!")
        print("请在您的出版物中正确引用Multiwfn，参考Multiwfn包中的\"How to cite Multiwfn.pdf\"文件")


def main():
    parser = argparse.ArgumentParser(description='RESP2电荷计算脚本')
    parser.add_argument('-i', '--input', required=True, help='输入分子结构文件(.xyz)')
    parser.add_argument('-c', '--charge', type=int, default=0, help='分子净电荷 (默认: 0)')
    parser.add_argument('-m', '--multiplicity', type=int, default=1, help='自旋多重度 (默认: 1)')
    parser.add_argument('-s', '--solvent', default='Water', help='溶剂名称 (默认: Water)')
    parser.add_argument('-o', '--output', default='resp2.chg', help='resp2电荷输出文件 (默认: resp2.chg)')
    parser.add_argument('-d', '--delta', type=float, default=0.5, help='RESP2液相charge权重系数 (默认: 0.5)')
    parser.add_argument('--orca', default='orca', help='ORCA程序路径')
    parser.add_argument('--orca_2mkl', default='orca_2mkl', help='orca_2mkl工具路径')
    parser.add_argument('--nprocs', type=int, default=8, help='CPU核心数 (默认: 8)')
    parser.add_argument('--maxcore', type=int, default=1000, help='每个核心内存(MB) (默认: 1000)')
    parser.add_argument('--keyword', default='! B3LYP/G D3 def2-TZVP def2/J RIJCOSX', help='ORCA计算关键词')
    parser.add_argument('--cleanup', action='store_true', help='清除中间文件')
    parser.add_argument('--equivcon', type=str,default=None, help='用户定义的equivalent constraints文件,格式： 1,2\n3,4,5\n\
    表示分别限制原子1/2同charge，原子3/4/5同charge，原子编号从1开始')
    
    args = parser.parse_args()
    
    # 显示参数
    print(f"输入文件: {args.input}")
    print(f"净电荷: {args.charge}")
    print(f"自旋多重度: {args.multiplicity}")
    print(f"溶剂: {args.solvent}")
    print(f"delta系数: {args.delta}")
    
    try:
        # 创建RESP对象并运行计算
        resp_calculator = RESP(
            input_file=args.input,
            charge=args.charge,
            multiplicity=args.multiplicity,
            solvent=args.solvent,
            delta=args.delta,
            orca_path=args.orca,
            orca_2mkl_path=args.orca_2mkl,
            nprocs=args.nprocs,
            maxcore=args.maxcore,
            keyword=args.keyword,
            output_file=args.output,
            equivcon=args.equivcon
        )
        
        resp_calculator.run(cleanup=args.cleanup)
        
    except KeyboardInterrupt:
        print("\n用户中断计算!")
        sys.exit(1)
    except Exception as e:
        print(f"\n计算过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()