'''
Author: Haotian Liang haotianliang10@gmail.com
Date: 2025-08-05 14:54:12
LastEditors: Haotian Liang haotianliang10@gmail.com
LastEditTime: 2025-08-06 18:54:26
'''
# from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
# from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
# from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
# from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig
# from experiments.hirol_reach.config import TrainConfig as HIROLReachTrainConfig
from experiments.fr3_reach.config import TrainConfig as FR3ReachTrainConfig

CONFIG_MAPPING = {
                # "ram_insertion": RAMInsertionTrainConfig,
                # "usb_pickup_insertion": USBPickupInsertionTrainConfig,
                # "object_handover": ObjectHandoverTrainConfig,
                # "egg_flip": EggFlipTrainConfig,
                # "hirol_reach": HIROLReachTrainConfig,
                "fr3_reach": FR3ReachTrainConfig,
               }