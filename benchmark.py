# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

#!/usr/bin/python
from utils import utilities, read_write_data, benchmark_argparser, run_benchmark_models
import sys
import os
import pandas as pd
import gc
import warnings
warnings.simplefilter("ignore")

def main():
    # Set Parameters
    arg_parser = benchmark_argparser()
    args = arg_parser.make_args()
    csv_file_path = args.csv_file_path
    model_path = args.model_dir
    # System Check
    system_check = utilities(jetson_devkit=args.jetson_devkit, gpu_freq=args.gpu_freq, dla_freq=args.dla_freq)
    system_check.close_all_apps()
    if system_check.check_trt():
        sys.exit()
    system_check.set_power_mode(args.power_mode, args.jetson_devkit)
    system_check.clear_ram_space()
    system_check.set_jetson_clocks()

    # Read CSV and Write Data
    benchmark_data = read_write_data(csv_file_path=csv_file_path, model_path=model_path)
    if args.all:
        latency_each_model =[]
        print("Running all benchmarks.. This will take at least 45 minutes...")
        for read_index in range (0,len(benchmark_data)):
            gc.collect()
            model = run_benchmark_models(csv_file_path=csv_file_path, model_path=model_path, benchmark_data=benchmark_data)
            download_err = model.execute(read_index=read_index)
            if not download_err:
                # Reading Results
                latency_fps, error_log = model.report()
                latency_each_model.append(latency_fps)
                # Remove engine and txt files
                if not error_log:
                    model.remove()
            del gc.garbage[:]
            system_check.clear_ram_space()
        benchmark_table = pd.DataFrame(latency_each_model, columns=['GPU (ms)', 'DLA0 (ms)', 'DLA1 (ms)', 'FPS', 'Model Name'], dtype=float)
        # Note: GPU, DLA latencies are measured in miliseconds, FPS = Frames per Second
        print(benchmark_table[['Model Name', 'FPS']])
        if args.plot:
            benchmark_data.plot_perf(latency_each_model)

    elif args.model_name == 'peoplenet':
        model = run_benchmark_models(csv_file_path=csv_file_path, model_path=model_path, benchmark_data=benchmark_data)
        download_err = model.execute(read_index=0)
        if not download_err:
            _, error_log = model.report()
            if not error_log:
                model.remove()

    elif args.model_name == 'action_recog_2d':
        model = run_benchmark_models(csv_file_path=csv_file_path, model_path=model_path, benchmark_data=benchmark_data)
        download_err = model.execute(read_index=1)
        if not download_err:
            _, error_log = model.report()
            if not error_log:
                model.remove()

    elif args.model_name == 'action_recog_3d':
        model = run_benchmark_models(csv_file_path=csv_file_path, model_path=model_path, benchmark_data=benchmark_data)
        download_err = model.execute(read_index=2)
        if not download_err:
            _, error_log = model.report()
            if not error_log:
                model.remove()

    elif args.model_name == 'dashcamnet':
        model = run_benchmark_models(csv_file_path=csv_file_path, model_path=model_path,  benchmark_data=benchmark_data)
        download_err = model.execute(read_index=3)
        if not download_err:
            _, error_log = model.report()
            if not error_log:
                model.remove()

    elif args.model_name == 'bodyposenet':
        model = run_benchmark_models(csv_file_path=csv_file_path, model_path=model_path,  benchmark_data=benchmark_data)
        download_err = model.execute(read_index=4)
        if not download_err:
            _, error_log = model.report()
            if not error_log:
                model.remove()

    elif args.model_name == 'lpr_us':
        model = run_benchmark_models(csv_file_path=csv_file_path, model_path=model_path,  benchmark_data=benchmark_data)
        download_err = model.execute(read_index=5)
        if not download_err:
            _, error_log = model.report()
            if not error_log:
                model.remove()

    

    system_check.clear_ram_space()
    
if __name__ == "__main__":
    main()
