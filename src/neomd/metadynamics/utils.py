def writehill(path, colvar, sigma_proj, sigma_ext, bias):
    with open(path, "w") as f:
        f.write(
            "#! FIELDS time pp.proj pp.ext sigma_pp.proj sigma_pp.ext height biasf\n"
        )
        f.write("#! SET multivariate false\n")
        f.write("#! SET kerneltype gaussian\n")
        for index, line in enumerate(colvar):
            time = index * 1e3 * 0.002
            f.write(
                f"{time:15} {line[0]:20.16f} {line[1]:20.16f}          {sigma_proj}           {sigma_ext} {line[3]:20.16f}            {bias}\n"
            )
