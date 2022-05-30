import os
import imp

def make_refiner(cfg):
    module = cfg.refiner_module
    path = cfg.refiner_path
    refiner = imp.load_source(module, path).Refiner()
    return refiner