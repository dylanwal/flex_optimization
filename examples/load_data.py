
import flex_optimization as fo


def main():
    file_name = "support_files//data_error_export_20220528-135216"
    recorder = fo.recorders.RecorderFull.load(file_name)

    vis = fo.OptimizationVis(recorder)
    fig = vis.plot_3d_vis()
    fig.show()


if __name__ == "__main__":
    main()
