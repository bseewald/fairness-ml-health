import numpy as np


def calc_confidence_interval(path):
    results = []

    with open(path, "r") as f:
        for line in f:
            results.append(float(line.strip()))

    s = np.std(results, ddof=1)
    s_over_n = s / np.sqrt(len(results))
    mean = np.mean(results)
    ic = (mean - 1.96 * s_over_n, mean + 1.96 * s_over_n)
    return ic


def main():
    # cindex confidence interval
    ic_women = calc_confidence_interval("files/cox-time/cindex/cindex_women.txt")
    ic_men = calc_confidence_interval("files/cox-time/cindex/cindex_men.txt")
    ic_white = calc_confidence_interval("files/cox-time/cindex/cindex_white.txt")
    ic_black = calc_confidence_interval("files/cox-time/cindex/cindex_black.txt")
    ic_women_black = calc_confidence_interval("files/cox-time/cindex/cindex_women_black.txt")
    ic_women_white = calc_confidence_interval("files/cox-time/cindex/cindex_women_white.txt")
    ic_men_black = calc_confidence_interval("files/cox-time/cindex/cindex_men_black.txt")
    ic_men_white = calc_confidence_interval("files/cox-time/cindex/cindex_men_white.txt")

    print(ic_women, ic_men)
    print(ic_white, ic_black)
    print(ic_women_black, ic_women_white)
    print(ic_men_black, ic_men_white)


if __name__ == "__main__":
    main()
