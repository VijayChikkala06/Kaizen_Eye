"""Run the converted backbone on the laptop (same .tflite the phone uses) -> [gh, gw, D] float32."""
import json
import os

import numpy as np


class TfliteEmbedder:
    def __init__(self, model_path):
        from ai_edge_litert.interpreter import Interpreter

        self.it = Interpreter(model_path=model_path)
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        self.o = self.it.get_output_details()[0]
        self.size = int(self.i["shape"][1])
        meta = os.path.splitext(model_path)[0] + ".json"
        self.meta = json.load(open(meta)) if os.path.exists(meta) else {}

    def __call__(self, img_u8):
        """img_u8: uint8 [size,size,3] RGB -> features [gh,gw,D]."""
        x = img_u8.astype(np.float32)[None]
        self.it.set_tensor(self.i["index"], x)
        self.it.invoke()
        return self.it.get_tensor(self.o["index"])[0].copy()
