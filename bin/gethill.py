import argparse
import sys

import numpy as np

"""
assume: 
    num of cvs == n
then: 
    line[0:n-1]==cv_1..cv_n
    line[n]==energy
    line[n+1]==current_height
    line[n+2:2n+1]==width_1..width_n
    line[2n+2]==biasFactor
    line[2n+3]==current_time
"""


def writehill(args):
    colvar = np.load(args.colvar)
    with open(args.path, "w") as f:
        nums = int((len(colvar[0]) - 4) / 2)
        cv_out = " ".join(["cv_" + str(i + 1) for i in range(nums)])
        sigma_out = " ".join(["sigma_cv_" + str(i + 1) for i in range(nums)])

        template = "#! FIELDS time " + cv_out + " " + sigma_out + " height biasf"
        period_temp = ""
        if not args.p_cvs is None:
            cv_ls = args.p_cvs.split(",")
            min_ls = args.min_cvs.strip(",").split(",")
            max_ls = args.max_cvs.strip(",").split(",")
            for i in range(len(cv_ls)):
                template += " min_{} max_{}".format(cv_ls[i], cv_ls[i])
                period_temp += "   {}   {}".format(min_ls[i], max_ls[i])

        f.write(template + "\n")
        f.write("#! SET multivariate false\n")
        f.write("#! SET kerneltype gaussian\n")

        for index, line in enumerate(colvar):
            time = line[2 * nums + 3]
            cv_out = ""
            sigma_out = ""
            for i in range(nums):
                cv_out += "{:20.16f} ".format(line[i])
                sigma_out += "          " + str(line[i + nums + 2]) + " "
            current_height = line[nums + 1]
            bias = line[2 * nums + 2]
            template = (
                f"{time:15.3f} "
                + cv_out
                + sigma_out
                + f"{current_height:20.16f}            {bias}"
            )
            f.write(template + period_temp + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get_hill handler")
    parser.add_argument("colvar", type=str, help="configuration file")
    parser.add_argument(
        "-period_cvs",
        dest="p_cvs",
        type=str,
        default=None,
        help="period cvs: cv_1,cv_2",
    )
    parser.add_argument(
        "-min_cvs", dest="min_cvs", type=str, default=None, help="min value of each cv"
    )
    parser.add_argument(
        "-max_cvs", dest="max_cvs", type=str, default=None, help="max value of each cv"
    )
    parser.add_argument(
        "-hills",
        "-hill",
        dest="path",
        type=str,
        default="./HILLS",
        help="path for saving HILLS file",
    )
    args = parser.parse_args()
    writehill(args)
