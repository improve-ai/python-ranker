import matplotlib.pyplot as plt


if __name__ == '__main__':

    x = [50000, 100000, 150000, 200000, 250000]
    est_data_load_times = [5, 8, 11, 14, 20.5]

    cyoff_total = [15, 23, 42, 58, 85]
    cyoff_runtime = [ct - cl for ct, cl in zip(cyoff_total, est_data_load_times)]

    # before last refactor
    cyon_total = [11, 13.5, 26, 32.5, 50]
    cyon_runtime = [ct - cl for ct, cl in zip(cyon_total, est_data_load_times)]

    full_cyon_total = [9.5, 11.5, 22.5, 28, 45]
    full_cyon_runtime = [ct - cl for ct, cl in zip(full_cyon_total, est_data_load_times)]

    speedup_ratio = [cn / cf for cn, cf in zip(cyon_runtime, cyoff_runtime)]
    full_cython_speedup_ratio = [cn / cf for cn, cf in zip(full_cyon_runtime, cyoff_runtime)]

    plt.plot(x, est_data_load_times, label='data load', linewidth=1, linestyle='--', color='red', marker='h')
    plt.plot(x, cyoff_runtime, label='plain python runtime (minus data load time)', linewidth=1, linestyle='--', color='blue', marker='h')
    plt.plot(x, cyon_runtime, label='only helper functions cythonized runtime (minus data load time)', linewidth=1, linestyle='--', color='green', marker='h')
    plt.plot(x, full_cyon_runtime, label='fully cythonic runtime (minus data load time)', linewidth=1, linestyle='--', color='black', marker='h')
    plt.xlabel('Number of decisions')
    plt.ylabel('Time [s]')
    plt.grid()
    plt.ylim(0, max(cyoff_runtime + cyon_runtime + est_data_load_times) * 1.2)
    plt.legend()
    plt.savefig('/home/kw/Projects/upwork/python-sdk/improveai/experiments/cython_speedup/raw_times.png')
    plt.clf()

    plt.plot(x, speedup_ratio, label='speedup ratio = (cythonized functions runtime) / (plain python runtime)', linewidth=1, linestyle='--', color='blue', marker='h')
    plt.plot(x, full_cython_speedup_ratio, label='speedup ratio = (fully cythonic runtime) / (plain python runtime)', linewidth=1, linestyle='--', color='green', marker='h')
    plt.xlabel('Number of decisions')
    plt.ylabel('speedup ratio [-]')
    plt.grid()
    plt.ylim(0, max(speedup_ratio) * 1.2)
    plt.legend()
    plt.savefig(
        '/home/kw/Projects/upwork/python-sdk/improveai/experiments/cython_speedup/cython_speedup.png')
    plt.clf()
