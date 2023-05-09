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
import os
import subprocess
import threading
import time

# Class for load, store, remove engine
class load_store_engine():
    def __init__(self, model_path, model_name, batch_size_gpu, batch_size_dla, num_devices, precision, model_input):
        self.model_path = model_path # Directory
        self.model_name = model_name # Model Name
        self.num_devices = num_devices # 3 if GPU+2DLA, 1 if GPU Only
        self.precision = precision # float16 or int8
        self.batch_size_gpu = batch_size_gpu # Batch Size for GPU
        self.batch_size_dla = batch_size_dla # Batch Size for DLA
        self. model_input = model_input # Input name of the model
        self.trt_process = []
        self.trt_set = set(["peoplenet", "dashcamnet"]) #set(["lpr_us", "bodyposenet", "action_recog_2d", "action_recog_3d"])

    def engine_gen(self):
        cmd = []
        model = []
        precision_cmd = str('--' + str(self.precision))
        for device_id in range(0, self.num_devices):
            if device_id == 1 or device_id == 2:
                self.device = 'dla'
                model_base_path = self._model2deploy()
                _model = str(self.model_name+"_"+self.device+"_bs"+str(self.batch_size_dla))
                dla_cmd = str('--useDLACore=' + str(device_id - 1))
                engine_CMD = str(
                    './trtexec' + " " + model_base_path + " " + precision_cmd + " " + " " + dla_cmd)
            else:
                self.device = 'gpu'
                model_base_path = self._model2deploy()
                _model = self.model_name+"_"+self.device
                engine_CMD = str(
                    './trtexec' + " " + model_base_path + " " + precision_cmd)
            cmd.append(engine_CMD)
            model.append(_model)
        
        
        return cmd, model

    def check_downloaded_models(self, model_name):
        model_name = str(self.model_name) + '.engine'
        model_file = os.path.join(self.model_path, model_name)
        if not os.path.isfile(model_file):
            print('Could Not find model file {} in {}\nPlease Download all model files'.format(model_files[e_id], self.model_path))
            return True
        return False

    def _model2deploy(self):
        if not self.model_name in self.trt_set:
            _model_input = str('--shapes='+str(self.model_input))
        else:
            if self.device == 'gpu':
                _model_input = str('--batch='+str(self.batch_size_gpu))
            else:
                _model_input = str('--batch='+str(self.batch_size_dla))
        return str(_model_input)


    def load_engine(self, _cmds, _models, load_output):
        
        load_engine_path = str('--loadEngine=' + str(os.path.join(self.model_path, _models)) + '.engine')
        avgruns_cmd = str('--avgRuns=100')+" "+'--duration=360'+" "+"--verbose"+" "#+"--threads" #+"--useSpinWait" #"--allowGPUFallback"
        cmd = str(_cmds)+" "+ avgruns_cmd + " " + str(load_engine_path)
        _trt_process = subprocess.Popen([cmd], cwd='/usr/src/tensorrt/bin/', shell=True, stdout=load_output,
                                      stderr=subprocess.STDOUT)
        self.trt_process.append(_trt_process)
        

    def load_all(self, commands, models):
        load_threads = []
        load_file_list = []
        for e_id in range(0, self.num_devices):
            model_device_name = str(models[e_id])
            if e_id > 0:
                model_device_name += "_"+str(e_id-1)
            load_file = os.path.join(self.model_path, model_device_name + '.txt')
            load_output = open(load_file, 'w')
            _load_threads = threading.Thread(target=self.load_engine(commands[e_id], models[e_id], load_output))
            load_threads.append(_load_threads)
            load_file_list.append(load_output)
            time.sleep(10)# Load memory
        # Start Threads 
        for lt in load_threads:
            lt.start()
        # Wait till threads are synchronize
        for lt in load_threads:
            lt.join()
        # Kill the subprocessess once complete
        for tp in self.trt_process:
            while tp.poll() == None:
                tp.poll()
            tp.kill()
        for flist in load_file_list:
            flist.close()

    def remove_engine(self, models):
        _txtout_path = str(str(os.path.join(self.model_path, models)) + '.txt')
        if os.path.isfile(_txtout_path):
            os.remove(_txtout_path)

    def remove_all(self, models):
        for e_id in range(0, self.num_devices):
            model_device_name = str(models[e_id])
            if e_id > 0:
                model_device_name += "_"+str(e_id-1)
            self.remove_engine(model_device_name)
