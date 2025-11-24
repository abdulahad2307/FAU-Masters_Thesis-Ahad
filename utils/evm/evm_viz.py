import matplotlib.pyplot as plt

def plot_openset_histograms(y_true, y_pred, scores, savepath=None):
    """
    Plots histogram of EVM scores for known and unknown true classes.
    """
    known = []
    unknown = []
    for yt, s in zip(y_true, scores):
        if yt == "unknown":
            unknown.append(s)
        else:
            known.append(s)
    plt.hist(known, bins=30, alpha=0.6, label="Known true")
    plt.hist(unknown, bins=30, alpha=0.6, label="Unknown true")
    plt.xlabel("EVM max probability / confidence")
    plt.ylabel("Frequency")
    plt.title("EVM Prediction Scores: Known vs Unknown Samples")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=120)
        print(f"Saved open set histogram to {savepath}")
    else:
        plt.show()
    plt.close()
