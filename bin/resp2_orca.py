#!/usr/bin/env python
"""v1 compatibility wrapper (one release): neomd.tools.orca Resp2Backend.

v1 argparse surface kept verbatim; flags map onto the Resp2Backend config.
"""
import argparse
import sys


def main(argv=None):
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
    parser.add_argument('--equivcon', type=str, default=None,
                        help='用户定义的equivalent constraints文件,格式： 1,2\n3,4,5\n')
    args = parser.parse_args(argv)

    from neomd.tools.orca import run as resp2_run
    resp2_run({
        "input": args.input, "charge": args.charge, "multiplicity": args.multiplicity,
        "solvent": args.solvent, "output": args.output, "delta": args.delta,
        "orca": args.orca, "orca_2mkl": args.orca_2mkl, "nprocs": args.nprocs,
        "maxcore": args.maxcore, "keyword": args.keyword, "cleanup": args.cleanup,
        "equivcon": args.equivcon,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
