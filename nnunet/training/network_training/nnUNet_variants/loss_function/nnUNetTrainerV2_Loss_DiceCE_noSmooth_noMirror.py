#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss, DC_and_CE_loss


class nnUNetTrainerV2_Loss_DiceCE_noSmooth_noMirror(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 0, 'do_bg': False}, {})

    def setup_DA_params(self):
        nnUNetTrainerV2.setup_DA_params(self)
        self.data_aug_params["do_mirror"] = False
        # from pprint import pprint
        # pprint(self.data_aug_params)

    def validate(self, do_mirroring: bool = False, use_sliding_window: bool = True,
                 step_size: int = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        do_mirroring = False
        super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                         save_softmax=save_softmax, use_gaussian=use_gaussian,
                         overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                         all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                         run_postprocessing_on_folds=run_postprocessing_on_folds)

