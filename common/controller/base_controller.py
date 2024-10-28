import logging
import time

import torch
from tqdm.auto import tqdm

from common.dataset.dataset_info import BasicDatasetInfo
from common.utils import GeneralUtils


class BaseController:
    def __init__(self):
        self._device = "cpu"
        self._num_workers = 4

        self._result_path = ""

        self._dataset_train: BasicDatasetInfo = None
        self._dataset_valid: BasicDatasetInfo = None
        self._dataset_test: BasicDatasetInfo = None
        self._train_data_feeder: torch.utils.data.DataLoader = None
        self._valid_data_feeder: torch.utils.data.DataLoader = None
        self._test_data_feeder: torch.utils.data.DataLoader = None

        self._epoch_cnt = -1
        self._cur_epoch = -1
        self._batch_size = -1
        self._cur_batch_idx = -1

        self._best_metric = float("inf")
        self._is_higher_metric_better = False

    def _change_models_phase(self, is_training: bool):
        pass

    def _train_batch(self, batch):
        pass

    @torch.no_grad()  # overriding 할 때 빼먹지 말고 다시 선언 해줘야 한다.
    def _validate_batch(self, batch):
        pass

    def _print_result(self, print_param):
        pass

    def _post_epoch_train(self, start_time, train_losses, train_metrics, valid_losses, valid_metrics) -> bool:
        pass  # return "Quit or Continue"

    def _post_epoch_test(self, start_time, test_losses, test_metrics):
        pass

    def _load_best_models_weights(self):
        pass

    def _save_best_models(self):
        pass

    def _save_last_models(self):
        pass

    def _set_models(self, param):
        pass

    def _set_opts_lrs(self, param):
        pass

    def _device_setup(self, gpu_idx):
        if gpu_idx < 0 or not torch.cuda.is_available():
            self._device = "cpu"
        elif torch.cuda.device_count() <= gpu_idx:
            self._device = "cuda"  # random assign
        else:
            self._device = f"cuda:{gpu_idx}"

        logging.info(f"Device : {self._device}")
        logging.info(f"cuDNN Available : {torch.backends.cudnn.is_available()}")
        logging.info(f"cuDNN Status : {torch.backends.cudnn.enabled}")
        logging.info(f"TensorFloat-32 : {torch.backends.cudnn.allow_tf32}")

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True

    def _get_data_feeder(self, dataset, shuffle) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self._batch_size, shuffle=shuffle, drop_last=False, pin_memory=False, num_workers=self._num_workers)

    def _set_metric_direction(self, is_higher_better):
        self._is_higher_metric_better = is_higher_better
        if is_higher_better:
            self._best_metric = float("-inf")
        else:
            self._best_metric = float("inf")

    def _set_early_stop(self, train_param):
        self._use_early_stop = train_param.get("use_early_stop")
        if self._use_early_stop:
            self._max_patience = train_param["patience"]
            self._min_epoch = train_param["min_epoch"]
        self._cur_patience = 0

    def _parse_train_param(self, train_param):
        self._device_setup(train_param["gpu_idx"])
        self._epoch_cnt = train_param["max_epoch"]
        self._batch_size = train_param["batch_size"]

        # data feeder
        self._train_data_feeder = self._get_data_feeder(self._dataset_train, True)
        self._valid_data_feeder = self._get_data_feeder(self._dataset_valid, False)

        self._result_path = train_param["print_param"]["result_path"]

        self._set_models(train_param)
        self._set_opts_lrs(train_param)

        self._set_early_stop(train_param)

    def _parse_test_param(self, test_param):
        self._device_setup(test_param["gpu_idx"])
        self._batch_size = test_param["batch_size"]

        # data feeder
        self._test_data_feeder = self._get_data_feeder(self._dataset_test, False)

        self._result_path = test_param["print_param"]["result_path"]

        self._set_models(test_param)
        self._load_best_models_weights()

    def _check_early_stop(self, monitoring_loss):
        if not self._use_early_stop:
            return

        if self._cur_epoch <= self._min_epoch:
            return

        if self._best_loss < monitoring_loss:
            self._cur_patience = self._cur_patience + 1
        else:
            self._best_loss = monitoring_loss
            self._cur_patience = 0

        if self._max_patience <= self._cur_patience:
            return True

        return False

    def _check_best_model(self, monitoring_metric):
        diff = self._best_metric - monitoring_metric
        if self._is_higher_metric_better:
            diff = -diff

        is_new_best = 0 < diff
        if is_new_best:
            self._best_metric = monitoring_metric

        return is_new_best

    def _pre_epoch(self, epoch):
        self._cur_epoch = epoch

    def _in_epoch(self, dataset, batch_process_function):
        epoch_metrics = []
        epoch_losses = []

        for batch_idx, batch in tqdm(enumerate(dataset), total=len(dataset), ascii=True):
            self._cur_batch_idx = batch_idx
            batch_loss, batch_metric = batch_process_function(batch)
            epoch_losses.append(batch_loss)
            epoch_metrics.append(batch_metric)

        return epoch_losses, epoch_metrics

    def do_train(self, dataset_train, dataset_valid, train_param):
        self._dataset_train = dataset_train
        self._dataset_valid = dataset_valid

        self._parse_train_param(train_param)
        logging.info(f"Train Batch Count : {len(self._train_data_feeder)}")

        GeneralUtils.set_seed(3)

        for epoch in range(0, self._epoch_cnt):
            start_time = time.time()
            self._pre_epoch(epoch)

            train_epoch_losses, train_epoch_metrics = self._in_epoch(self._train_data_feeder, self._train_batch)

            valid_epoch_losses = valid_epoch_metrics = None
            if self._dataset_valid is not None and 0 < len(self._valid_data_feeder):
                valid_epoch_losses, valid_epoch_metrics = self._in_epoch(self._valid_data_feeder, self._validate_batch)

            if self._post_epoch_train(start_time, train_epoch_losses, train_epoch_metrics, valid_epoch_losses, valid_epoch_metrics):
                break

        self._save_last_models()
        self._load_best_models_weights()

        logging.info("Training is Finish. Now printing result...")
        with torch.no_grad():
            self._print_result(train_param["print_param"])
        logging.info("Printing result is Finish.")

    def do_test(self, dataset_test, test_param):
        self._dataset_test = dataset_test

        self._parse_test_param(test_param)
        
        GeneralUtils.set_seed(2)

        start_time = time.time()
        with torch.no_grad():
            # self._on_epoch_start()
            test_loss, test_metric = self._in_epoch(self._test_data_feeder, self._validate_batch)
            self._post_epoch_test(start_time, test_loss, test_metric)

            self._print_result(test_param["print_param"])
        logging.info("Testing is Finish.")
