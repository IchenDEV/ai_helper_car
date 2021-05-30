#!/usr/bin/env python
"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore


class FaceNet(object):
    def __init__(self, model_xml, device="CPU"):

        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.device = device
        # Plugin initialization for specified device and load extensions library if specified
        log.info("Creating Inference Engine")
        self.ie = IECore()
        # Read IR
        log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        self.net = IENetwork(model=model_xml, weights=model_bin)

        if "CPU" in device:
            supported_layers = self.ie.query_network(self.net, "CPU")
            not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(device, ', '.join(not_supported_layers)))
                log.error(
                    "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                    "or --cpu_extension command line argument")
                sys.exit(1)

        assert len(self.net.inputs.keys()) == 1, "Sample supports only single input topologies"
        assert len(self.net.outputs) == 1, "Sample supports only single output topologies"

        log.info("Preparing input blobs")
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        self.net.batch_size = 1
        log.info("Batch size is {}".format(self.net.batch_size))

        # Loading model to the plugin
        log.info("Loading model to the plugin")
        self.exec_net = self.ie.load_network(network=self.net, device_name=self.device)

    def get_feature(self, image):
        # Read and pre-process input images
        n, c, h, w = self.net.inputs[self.input_blob].shape

        if image.shape[:-1] != (h, w):
            log.warning("Image  is resized from {} to {}".format(image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW

        # Start sync inference
        log.info("Starting inference in synchronous mode")
        res = self.exec_net.infer(inputs={self.input_blob: image})

        # Processing output blob
        log.info("Processing output blob")
        res = res[self.out_blob]
        norm = np.linalg.norm(res, axis=1, keepdims=True)
        feature = res / norm
        return feature
